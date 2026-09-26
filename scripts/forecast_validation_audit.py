"""Reproduce F2 split, checkpoint, comparator and solver-reference audits."""
import argparse
import json
import sys

import numpy as np
import torch

from forecast_validate import ROOT, OUT, from_d1, digest, write_json
from forecast_application import Ensemble, initial_nfw, interpolate
from e9_inference import EnsembleEmulator, R_OBS, solver_profile_at
from statecond import data
from train import split as production_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    full_data = from_d1(ROOT/'data/d1/train.h5')
    full = data.split(len(full_data['u']))
    assert all(np.array_equal(a, b) for a, b in
               zip(full, production_split(len(full_data['u']))))
    terminal = from_d1(ROOT/'data/d1/train.h5', True)
    legacy = terminal['source_index'][data.split(len(terminal['u']))[0]]
    fresh = from_d1(ROOT/'data/d1/forecast_fresh.h5')
    identities = {np.concatenate([p, u.ravel()]).tobytes()
                  for p, u in zip(full_data['par'], full_data['u0'])}
    overlap = sum(np.concatenate([p, u.ravel()]).tobytes() in identities
                  for p, u in zip(fresh['par'], fresh['u0']))
    assert overlap == 0
    audit = dict(full_split_matches_production=True,
                 full_train_count=len(full[0]), full_validation_count=len(full[1]),
                 full_test_count=len(full[2]), fresh_full_input_overlap=overlap,
                 legacy_terminal_clock_train_overlap_full_validation=int(np.isin(legacy, full[1]).sum()),
                 legacy_terminal_clock_train_overlap_full_test=int(np.isin(legacy, full[2]).sum()),
                 new_clocks={})
    for seed in range(3):
        clock_path = OUT/'application'/f'clock_seed{seed}.json'
        rec = json.loads(clock_path.read_text())
        assert np.array_equal(rec['train_index'], full[0])
        assert rec['checkpoint_sha256'] == digest(clock_path.with_suffix('.pt'))
        assert rec['data_sha256'] == digest(ROOT/'data/sc/sidm_full.h5')
        field = OUT/'application'/f'sidm_full_s{seed}'
        cfg = json.loads((field/'config.json').read_text())
        assert cfg['clock_sha256'] == rec['checkpoint_sha256']
        assert np.array_equal(cfg['train_index'], full[0])
        assert np.array_equal(cfg['test_index'], full[2])
        assert json.loads((field/'training.json').read_text())['history'][-1]['epoch'] == 200
        audit['new_clocks'][str(seed)] = dict(validation_overlap=0, test_overlap=0)
    for system in ('sidm', 'rd', 'ks', 'film', 'burgers'):
        confirmation = json.loads((OUT/'confirmation'/f'{system}.json').read_text())
        assert confirmation['fresh_sha256'] == digest(ROOT/'data/sc/forecast_fresh'/f'{system}.h5')
        for path, expected in confirmation['checkpoint_hashes'].items():
            assert digest(ROOT/path) == expected, path
        for seed in range(3):
            control = OUT/'controls'/f'{system}_s{seed}'
            cfg = json.loads((control/'config.json').read_text())
            assert cfg['data_sha256'] == digest(ROOT/'data/sc'/f'{system}.h5')
            assert json.loads((control/'training.json').read_text())['history'][-1]['epoch'] == 200
    production = Ensemble('production', args.device)
    original = EnsembleEmulator([ROOT/'results/runs'/f'gravop_s{s}'/'model.pt'
                                 for s in range(5)], device=args.device)
    params = np.array([[-1., .2, 1.], [-.3, .7, 1.], [.2, 1.2, 1.]], dtype='f4')
    y, ratio = production.predict(np.repeat(initial_nfw()[None], 3, 0), params, 20.)
    z, _, old_ratio, _ = original.predict(params[:, 0], params[:, 1], 20.)
    delta = float(np.max(np.abs(interpolate(y[:, 0], R_OBS)-z)))
    delta_ratio = float(np.max(np.abs(ratio-old_ratio)))
    # Float32 CPU/GPU arithmetic need not be bitwise identical. 3e-5 dex is
    # <0.007% density, much smaller than the measured application errors.
    assert delta < 3e-5 and delta_ratio < 1e-6, (delta, delta_ratio)
    audit.update(production_prediction_max_abs_dex=delta,
                 production_ratio_max_abs=delta_ratio,
                 production_numerical_tolerance_dex=3e-5,
                 production_parity_note='Three NFW parameter pairs at t=20; float32 numerical agreement.')
    write_json(OUT/'application/implementation_audit.json', audit)
    mock = json.loads((ROOT/'results/e9/mock.json').read_text())
    rows = []
    for name in ('grid_logpost.npz', 'grid_zoom.npz'):
        grid = np.load(ROOT/'results/e9'/name)
        i, j = np.unravel_index(np.argmax(grid['logpost']), grid['logpost'].shape)
        pred, _ = solver_profile_at(10.**grid['lsig'][i], 10.**grid['lw'][j], t_obs=mock['t_obs'])
        lp = -.5*np.sum((pred-np.asarray(mock['lrho_obs']))**2/mock['noise_dex']**2)
        delta = float(abs(lp-grid['logpost'][i, j]))
        assert delta < 1e-6, (name, delta)
        rows.append(dict(grid=name, index=[int(i), int(j)],
                         stored_logpost=float(grid['logpost'][i, j]),
                         recomputed_logpost=float(lp), abs_difference=delta))
    write_json(OUT/'application/solver_grid_audit.json', rows)
    write_json(OUT/'execution_provenance.json', dict(
        python=sys.version, torch=torch.__version__, numpy=np.__version__,
        audit_device=args.device,
        gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
        application_clock_training_devices=['cpu']*3,
        application_field_training_devices=['cuda:1']*3,
        source_sha256={p: digest(ROOT/p) for p in (
            'scripts/forecast_validate.py', 'scripts/forecast_application.py',
            'scripts/forecast_validation_audit.py', 'scripts/forecast_validation_suite.py',
            'scripts/sc_forecast.py', 'src/statecond/forecast.py')},
        note='Training devices describe the original F2 run; audit_device describes this audit.'))
    print('F2 audit passed: splits, identities, hashes, completed budgets, production parity, solver grid modes.')


if __name__ == '__main__':
    main()
