"""Independent F7 arithmetic audit, temporal validation and final report."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing

import numpy as np
import torch

import forecast_conditional as f6
from forecast_posterior_refine import DATA, OUT, ROOT, arrays, column_path, prepare, worker_init
from forecast_numerics import refined_row
from forecast_numerics_pilot import compare
from forecast_calibrate import LO, HI, NOISE_DEX, write_npz
from forecast_validate import digest, write_json
from gravop.conditional_refinement import OuterMesh


def main():
    torch.set_num_threads(1)
    frozen, obs = prepare()
    snapshot = json.loads((OUT/'source_snapshot.json').read_text())
    for name, expected in snapshot['files'].items():
        assert digest(ROOT/name) == expected, f'F7 archived dependency changed: {name}'
    progress = json.loads((OUT/'progress.json').read_text())
    if not progress['complete']:
        raise RuntimeError('F7 is still running')
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
            lp.append(-.5*np.square((profile[1]-obs['observations'][0])/NOISE_DEX).sum() if valid[1] else -np.inf)
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
    test_log = ROOT/'logs/f7_tests.log'
    if test_log.exists():
        result['test_log'] = dict(path=str(test_log.relative_to(ROOT)), sha256=digest(test_log),
                                  summary=test_log.read_text().strip().splitlines()[-1])
    write_json(OUT/'validation.json', result)
    report(result)
    print('F7 audit/report complete; overall passed', result['passed'], flush=True)


def report(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = d['progress']['history']
    if history:
        fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
        for i, name in enumerate(('solver', 'candidate')):
            ax.semilogy([r['submissions'] for r in history],
                        [np.max(r['relative_error'][i]) for r in history], 'o-', label=name)
        for threshold in (.02, .005, .00125):
            ax.axhline(threshold, color='gray', ls='--', lw=.7)
            ax.text(.99, threshold, f'{threshold:g}', ha='right', va='bottom', fontsize=8,
                    transform=ax.get_yaxis_transform())
        ax.set(xlabel='New level-1 solver submissions', ylabel='Maximum propagated relative error')
        ax.legend()
        fig.savefig(OUT/'convergence.png', dpi=170)
        fig.savefig(OUT/'convergence.pdf')
        plt.close(fig)
    lines = ['# F7 — posterior convergence repair (2026-09-10)', '',
        f"**Overall numerical and temporal validation passed: {d['passed']}.** One reused diagnostic case at time 400; this is not population calibration. Models, production and manuscript remain unchanged.", '',
        'Protocol: [FORECAST_POSTERIOR_REFINE_PROTOCOL.md](FORECAST_POSTERIOR_REFINE_PROTOCOL.md). The failed [F6 experiment](FORECAST_CONDITIONAL_RESULTS_2026-09-10.md) is preserved.', '',
        'The revised outer tree uses fixed base bounds and integer dyadic descendants, so inherited nodes retain exact cache identities. Observed parameter-CDF, likelihood-CDF and evidence changes guide initial branch refinement. Complete verified inner partitions are reused at tighter tolerances. Already-required survey and quadrature evaluations are batched through a shared solver pool; the full prior, both survival supports and direct likelihoods are retained.', '',
        'The design follows the use of explicit difficult-feature breakpoints in [QUADPACK](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html), while retaining custom positive vector rules and propagated inner errors. A finite discovery survey is not a proof against arbitrary hidden modes or support islands.', '',
        f"Stop: `{d['progress']['reason']}`. Reserved new level-1 submissions: {d['progress']['submissions']}; recorded completed evaluations: {d['completed_new_level1_evaluations']}. The separately frozen ceiling is 80,000 submissions, 1,024 per column. Preflight cost accounting increased the unused preliminary ceiling before any F7 solver submissions; those preliminary records are preserved.", '',
        'Execution started with 24 solver workers and 12 column threads, then resumed from durable caches with 24 column threads after a concurrency check. The reservation ledger retains 18 submissions without recorded completed evaluations across that restart; they were not refunded. See `concurrency_restart.json` and `process.json`. Numerical sources and acceptance gates stayed fixed.', '',
        '## Convergence checks', '',
        '| Stage | Panels | Columns | New submissions | Solver maximum error | Candidate maximum error |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in history:
        lines.append(f"| {r['stage']} | {r['panels']} | {r['columns']} | {r['submissions']} | {np.max(r['relative_error'][0]):.7g} | {np.max(r['relative_error'][1]):.7g} |")
    lines += ['', '![Convergence](../results/forecast_posterior_refine/convergence.png)', '',
        'The unchanged targets are 0.02, 0.005 and optional 0.00125; inner tolerances are one tenth as large. Every outer panel is bisected between stages. Both methods must satisfy mean shifts <=0.1 final SD, relative 90%-width changes <=0.1 and all three CDF changes <=0.02. Marginal widths remain positive-node approximations and are explicitly checked for stability.', '']
    if d.get('convergence'):
        lines += ['| Method | Largest mean shift (SD) | Largest relative width change | Largest CDF change | Passed |',
                  '|---|---:|---:|---:|---|']
        for m, r in d['convergence'].items():
            lines.append(f"| {m} | {np.max(r['mean_shift_sd']):.7g} | {np.max(r['relative_width_change']):.7g} | {np.max(r['cdf_change']):.7g} | {r['passed']} |")
        lines.append('')
    if d.get('latest'):
        lines += ['| Method | Mean (log sigma0 / log w) | 90% widths | CDFs (sigma0 / w / solver likelihood) | TV to solver |',
                  '|---|---|---|---|---:|']
        fmt = lambda a: ' / '.join(f'{v:.6g}' for v in a)
        for m, r in d['latest'].items():
            lines.append(f"| {m} | {fmt(r['mean'][0])} | {fmt(r['width90'][0])} | {fmt(r['cdf'][0])} | {r['tv'][0]:.6g} |")
        lines.append('')
    if d['temporal']:
        r = d['temporal']
        lines += [f"Temporal check: {r['n']} selected peaks, {r['new_evaluations']} new level-2 evaluations, maximum profile change {r['max_profile_dex']:.7g} dex, support disagreements `{r['support_disagreements']}`, passed **{r['passed']}**. Selection was saved before new evaluations.", '']
    lines += ['## Audit and interpretation', '',
        'The audit independently reproduces positive full-prior weights, cached solver likelihoods, vector integrals, propagated errors, means, widths and CDFs, and rechecks frozen sources/data/checkpoints. When tolerance stages finish, their recorded comparison is recomputed. Solver reservations are durable and exclusive-run locking prevents duplicate drivers.', '',
        f"Test result: `{d.get('test_log', {}).get('summary', 'not recorded')}`. The manuscript-count check remains deferred. Analytic tests include a complete two-method outer integration of curved Gaussian ridges with exact conditional moments/CDFs and independent reference integrals, reuse of verified inner partitions, exact outer-node identities and concurrent budget reservations. Concurrency checks matched serial frozen-candidate outputs bit-for-bit for 36 predictions with 12 threads and 72 predictions with 24 threads.", '']
    if d['passed']:
        lines += ['The repaired method clears the prescribed numerical gates for this case. This supports extending the same validation to other difficult observations and times before fresh population calibration. It does not establish population calibration or justify production replacement.', '']
    else:
        lines += ['The recorded configuration has not cleared the complete numerical/temporal gate. Retain the failure and diagnose the remaining component before broader calibration; do not interpret a cap or partial rule as convergence.', '']
    lines += ['```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_posterior_refine.py --workers 24 --columns 24',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_posterior_refine_report.py',
        '```', '', 'Artifacts: `results/forecast_posterior_refine/`; caches: `data/d1/forecast_posterior_refine/`; logs: `logs/f7_*`. Earlier failed experiments remain unchanged.', '']
    (ROOT/'docs/FORECAST_POSTERIOR_REFINE_RESULTS_2026-09-10.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
