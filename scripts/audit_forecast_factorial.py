"""Audit completed F10 raw records and the preserved training source snapshot."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from forecast_factorial import ARMS, SYSTEMS, OUT, FactorialOperator, prepare, errors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_verification_labels(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = [d for d in node.decorator_list
                                   if ast.unparse(d) != 'pytest.mark.verification']
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='Also replay all saved validation predictions on this device')
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((OUT/'manifest.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text())
    assert summary['complete']
    annotations = []
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        for name, expected in manifest['files'].items():
            if Path(name).suffix in {'.py', '.md'}:
                original = z.read(name)
                assert hashlib.sha256(original).hexdigest() == expected, name
            if digest(ROOT/name) != expected:
                # The full repository gate requires classification decorators.
                # They are added only after this experiment finishes; compare
                # the full AST to allow no other executable source change.
                assert name == 'tests/test_forecast_factorial.py', name
                assert without_verification_labels((ROOT/name).read_text()) == without_verification_labels(original.decode())
                annotations.append(dict(path=name, original_sha256=expected,
                                        current_sha256=digest(ROOT/name), change='verification decorators only'))
    records = []; parameter_counts = {}
    for system in SYSTEMS:
        values = {}
        if args.device:
            _, _, u, u0, par, times, _, va, _ = prepare(system, args.device)
        for arm in ARMS:
            path = OUT/f'{system}_{arm}_s0'
            config = json.loads((path/'config.json').read_text())
            training = json.loads((path/'training.json').read_text())
            result = json.loads((path/'validation.json').read_text())
            assert config['manifest_sha256'] == result['manifest_sha256'] == digest(OUT/'manifest.json')
            assert config['system'] == system and config['arm'] == arm and config['seed'] == 0
            assert config['epochs'] == training['history'][-1]['epoch'] == 200
            assert [r['epoch'] for r in training['history']] == [1]+list(range(20,201,20))
            assert not config['test_evaluated'] and not result['test_evaluated']
            assert not set(config['train_index']) & set(config['validation_index'])
            assert set(config['cpu_affinity']) <= set(range(6))|set(range(12,18))
            assert digest(path/'model.pt') == result['checkpoint_sha256']
            ck = torch.load(path/'model.pt', map_location='cpu', weights_only=False)
            assert ck['config'] == config and ck['validation'] == result['validation']
            assert ck['epoch'] == result['selected_epoch']
            best = min(training['history'], key=lambda r:r['validation'])
            assert best['epoch'] == ck['epoch']
            score = result['validation']['nrmse']
            np.testing.assert_allclose(score, np.mean(result['validation']['per_trajectory']), rtol=1e-6)
            np.testing.assert_allclose(score, np.mean(result['validation']['per_time']), rtol=1e-6)
            assert score == best['validation'] == summary['systems'][system]['arms'][arm]['nrmse']
            count = (training['trainable_parameters'], training['total_parameters'])
            assert count == parameter_counts.setdefault(system, count)
            values[arm] = score
            if args.device:
                model = FactorialOperator(arm, **config['model']).to(args.device)
                model.load_state_dict(ck['state'])
                replay = errors(model, u, u0, par, times, va)
                np.testing.assert_allclose(replay.mean(1), result['validation']['per_trajectory'], rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(replay.mean(0), result['validation']['per_time'], rtol=1e-5, atol=1e-7)
                del model
            records.append(dict(system=system, arm=arm, nrmse=score, seconds=training['seconds'],
                                epoch=ck['epoch'], parameters=count))
        interaction = values['full']*values['direct']/(values['features_only']*values['gate_only'])
        np.testing.assert_allclose(interaction, summary['systems'][system]['multiplicative_interaction'], rtol=1e-14)
    record = dict(passed=True, fits=len(records), records=records,
                  checkpoint_validation_replayed=bool(args.device), replay_device=args.device,
                  postrun_test_annotations=annotations, source_snapshot_sha256=digest(OUT/'source_snapshot.zip'),
                  manifest_sha256=digest(OUT/'manifest.json'),
                  note='Original training source hashes retained; no numerical source or manifest hash repaired.')
    (OUT/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    if args.device:
        (OUT/'audit_replayed.json').write_text(json.dumps(record,indent=2)+'\n')
    print('F10 audit PASS:',len(records),'fits; original sources, paired configuration, checkpoints and report verified.')


if __name__ == '__main__':
    main()
