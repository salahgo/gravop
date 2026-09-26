"""Independent F8 arithmetic audit, temporal validation and final report."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing

import numpy as np
import torch

import forecast_conditional as f6
import forecast_final_case as f8
from forecast_final_case import ROOT, arrays, worker_init

def column_path(y):
    return f8.column_path(y)
from forecast_numerics import refined_row
from forecast_numerics_pilot import compare
from forecast_calibrate import LO, HI, NOISE_DEX, write_npz
from forecast_validate import digest, write_json
from gravop.conditional_refinement import OuterMesh


def main():
    torch.set_num_threads(1)
    frozen, obs = f8.prepare()
    snapshot = json.loads((f8.BASE_OUT/'config.json').read_text())
    for name, expected in snapshot['sources'].items():
        assert digest(ROOT/name) == expected, f'F8 archived dependency changed: {name}'
    progress = json.loads((OUT/'progress.json').read_text())
    if not progress['complete']:
        raise RuntimeError('F8 is still running')
    result = dict(config=frozen, progress=progress, audit={}, temporal=None, passed=False)
    if (DATA/'latest_rule.npz').exists():
        rule = arrays(DATA/'latest_rule.npz')
        stage = int(rule['stage'])
        mesh = OuterMesh(rule['roots'], rule['cells'])
        columns, cache = {}, {}
        for y in mesh.required():
            path = column_path(y)/f'stage{stage}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['converged'] and digest(path) == meta['sha256']
            columns[y] = arrays(path)
            rows = arrays(column_path(y)/'nodes.npz')
            cache[y] = {float(x): (rows['profiles'][i], rows['valid'][i]) for i, x in enumerate(rows['x'])}
        panels = mesh.assess(columns, obs['params'][0, 1])
        value, error = sum(p['value'] for p in panels), sum(p['error'] for p in panels)
        np.testing.assert_allclose(value, rule['integrated'], rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(error, rule['error'], rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(error/value[:, :1], progress['history'][-1]['relative_error'], rtol=1e-12)
        assert (rule['weights'] > 0).all()
        np.testing.assert_allclose(rule['weights'].sum(), np.prod(HI-LO), rtol=0, atol=1e-12)
        lp = []
        for x, y in rule['theta']:
            profile, valid = cache[y][float(x)]
            lp.append(-.5*np.square((profile[int(obs['time_index'][0])]-obs['observations'][0])/NOISE_DEX).sum() if valid[int(obs['time_index'][0])] else -np.inf)
        np.testing.assert_allclose(lp, rule['lp'][:, 0], atol=1e-10, rtol=1e-12)
        reproduced, _ = f6.summarize_rule(mesh.intervals(), panels, columns, obs)
        result['latest'] = {}
        for i, method in enumerate(('solver', 'candidate')):
            saved = arrays(OUT/f'latest_{method}.npz')
            for name in ('mean', 'sd', 'width90', 'cdf', 'tv'):
                np.testing.assert_allclose(reproduced[method][name], saved[name], atol=1e-12, rtol=1e-12)
            result['latest'][method] = {k: v.tolist() for k, v in saved.items()}
        result['audit'] = dict(frozen_source_data_models=True, positive_full_prior_weights=True,
            reproduced_integrals_errors=True, direct_likelihoods_agree=True, reproduced_summaries=True)
        if (OUT/'convergence.json').exists():
            recorded = json.loads((OUT/'convergence.json').read_text())
            assert stage >= 1
            actual = {m: compare(arrays(OUT/f'stage{stage-1}_{m}.npz'), arrays(OUT/f'stage{stage}_{m}.npz'))
                      for m in ('solver', 'candidate')}
            assert recorded == actual
            result['convergence'] = actual
        if progress['numerical_passed']:
            assert result.get('convergence') and all(v['passed'] for v in result['convergence'].values())
            assert np.max(error/value[:, :1]) <= frozen['targets'][stage]
            result['audit']['successive_gates_reproduced'] = True
        q = rule['weights']*np.exp(rule['lp'][:, 0]-np.max(rule['lp'][:, 0]))
        q /= q.sum()
        ys = np.unique(rule['theta'][:, 1])
        mass = np.array([q[rule['theta'][:, 1] == y].sum() for y in ys])
        chosen = np.unique(ys[np.searchsorted(np.cumsum(mass), [.1, .3, .5, .7, .9])])
        params = []
        for y in chosen:
            meta = json.loads((column_path(y)/f'stage{stage}.json').read_text())
            modes = [m for m in meta['discovery']['modes'] if m['method'] == 0]
            params.append([max(modes, key=lambda m: m['loglike'])['x'], float(y)])
        params = np.array(params)
        selection = dict(params=params.tolist(), rule_sha256=digest(DATA/'latest_rule.npz'),
                         selection='solver marginal posterior-mass quantiles .1/.3/.5/.7/.9, deduplicated')
        path = OUT/'temporal_selection.json'
        if path.exists():
            assert json.loads(path.read_text()) == selection
        else:
            write_json(path, selection)
        strict_path = DATA/'mode_refinement.npz'
        if strict_path.exists():
            strict = arrays(strict_path)
            np.testing.assert_array_equal(strict['params'], params)
        else:
            old = arrays(f6.DATA/'mode_refinement.npz')
            profiles = np.full((len(params), 3, 20), np.nan)
            valid = np.zeros((len(params), 3), bool)
            status = np.full(len(params), -1)
            pending, reused = [], []
            for i, p in enumerate(params):
                matches = np.flatnonzero(np.all(old['params'] == p, axis=1))
                if len(matches):
                    j = matches[0]
                    profiles[i], valid[i], status[i] = old['profiles'][j], old['valid'][j], old['status'][j]
                    reused.append(i)
                else:
                    pending.append((i, p, 2))
            if pending:
                with ProcessPoolExecutor(max_workers=len(pending), initializer=worker_init,
                        mp_context=multiprocessing.get_context('spawn')) as pool:
                    rows = list(pool.map(refined_row, pending))
                failures = [r[-1] for r in rows if r[-1]]
                if failures:
                    write_json(OUT/'temporal_failures.json', failures)
                    raise RuntimeError(failures)
                for i, profile, mask, st, end, failure in rows:
                    profiles[i], valid[i], status[i] = profile, mask, st
            strict = dict(params=params, profiles=profiles, valid=valid, status=status,
                          reused_indices=np.array(reused, dtype=int), new_evaluations=len(pending))
            write_npz(strict_path, **strict)
        baseline, valid = [], []
        for x, y in params:
            profile, mask = cache[y][float(x)]
            baseline.append(profile)
            valid.append(mask)
        baseline, valid = np.array(baseline), np.array(valid)
        common = valid & strict['valid']
        disagreements = np.argwhere(valid != strict['valid']).tolist()
        maximum = float(np.max(abs(strict['profiles'][common]-baseline[common])))
        result['temporal'] = dict(n=len(params), new_evaluations=int(strict['new_evaluations']),
            max_profile_dex=maximum, support_disagreements=disagreements,
            sha256=digest(strict_path), passed=bool(not disagreements and maximum <= .00434))
        result['passed'] = bool(progress['numerical_passed'] and result['temporal']['passed'])
        result['panel_errors'] = [dict(bounds=p['y'][[0, -1]].tolist(),
            total_relative=(p['error']/value[:, :1]).tolist(),
            inner_relative=(p['inner_error']/value[:, :1]).tolist()) for p in panels]
    result['completed_new_level1_evaluations'] = sum(int(arrays(p)['new_count']) for p in (DATA/'columns').glob('*/nodes.npz'))
    result['report_source_sha256'] = digest(__file__)
    test_log = ROOT/'logs/f8_tests.log'
    if test_log.exists():
        result['test_log'] = dict(path=str(test_log.relative_to(ROOT)), sha256=digest(test_log),
                                  summary=test_log.read_text().strip().splitlines()[-1])
    write_json(OUT/'validation.json', result)
    print('F8 audit/report complete; overall passed', result['passed'], flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=int, required=True)
    args = p.parse_args()
    DATA = f8.DATA = f8.BASE_DATA/f'case{args.case}'
    OUT = f8.OUT = f8.BASE_OUT/f'case{args.case}'
    main()
