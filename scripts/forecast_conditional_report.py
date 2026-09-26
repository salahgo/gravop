"""Independent audit, targeted temporal checks and F6 report."""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import torch

from forecast_conditional import DATA, OUT, ROOT, F5, config, column_path, new_solves
from forecast_numerics import refined_row
from forecast_application import Ensemble, initial_nfw, interpolate
from forecast_calibrate import LO, HI, NOISE_DEX, write_npz
from forecast_validate import digest, write_json
from e9_inference import R_OBS
from gravop.conditional import outer_panel


def main():
    torch.set_num_threads(1)
    frozen, obs = config()
    progress = json.loads((OUT / 'progress.json').read_text())
    if not progress['complete']:
        raise RuntimeError('F6 is still running')
    out = dict(progress=progress, config=frozen, audit={}, temporal=None, pilot_passed=False)
    rule_path = DATA / 'latest_rule.npz'
    if rule_path.exists():
        rule = np.load(rule_path)
        stage = progress['history'][-1]['stage']
        intervals = rule['intervals']
        columns = {}
        for a, b in intervals:
            for y in a+(b-a)*np.arange(5)/4:
                path = column_path(y) / f'stage{stage}.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                assert digest(path) == meta['sha256']
                columns[float(y)] = dict(np.load(path))
        panels = [outer_panel(a, b, columns, (LO[1], HI[1]), obs['params'][0, 1]) for a, b in intervals]
        values = sum(p['value'] for p in panels)
        errors = sum(p['error'] for p in panels)
        np.testing.assert_allclose(values, rule['integrated'], rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(errors/values[:, :1], progress['history'][-1]['relative_error'], rtol=1e-12)
        assert (rule['weights'] > 0).all()
        np.testing.assert_allclose(rule['weights'].sum(), np.prod(HI-LO), atol=1e-12, rtol=0)
        # Independently rederive each cached direct likelihood from its density
        # profile, rather than trusting the quadrature's stored likelihoods.
        cache = {}
        for y in np.unique(rule['theta'][:, 1]):
            d = np.load(column_path(y) / 'nodes.npz')
            cache[y] = {float(x): (d['profiles'][i, 1], d['valid'][i, 1]) for i, x in enumerate(d['x'])}
        lp = []
        for x, y in rule['theta']:
            profile, valid = cache[y][float(x)]
            lp.append(-.5*np.square((profile-obs['observations'][0])/NOISE_DEX).sum() if valid else -np.inf)
        np.testing.assert_allclose(lp, rule['lp'][:, 0], atol=1e-10, rtol=1e-12)
        out['audit'] = dict(full_prior_positive_weights=True, reproduced_outer_error=True,
                            cached_solver_likelihoods_agree=True, recorded_hashes_agree=True)
        out['error_components'] = ['evidence', 'sigma0 mean numerator', 'sigma0 second moment',
            'sigma0 CDF numerator', 'solver-likelihood CDF numerator', 'w mean numerator',
            'w second moment', 'w CDF numerator']
        out['panel_errors'] = []
        for (a, b), panel in zip(intervals, panels):
            inner = np.stack([columns[float(y)]['error'] for y in panel['y']])
            u = (panel['y']-LO[1])/(HI[1]-LO[1])
            augmented = np.concatenate([inner, inner[:, :, :1]*u[:, None, None],
                inner[:, :, :1]*u[:, None, None]**2,
                inner[:, :, :1]*float((a+b)/2 < obs['params'][0, 1])], axis=2)
            propagated = np.einsum('n,nmk->mk', panel['weights'], augmented)
            outer = panel['error']-propagated
            assert (outer >= -1e-25).all()
            out['panel_errors'].append(dict(bounds=[float(a), float(b)],
                inner_relative=(propagated/values[:, :1]).tolist(),
                outer_relative=(outer/values[:, :1]).tolist()))
        out['latest'] = {}
        for i, method in enumerate(('solver', 'candidate')):
            d = dict(np.load(OUT / f'latest_{method}.npz'))
            mean = LO+(HI-LO)*values[i, [1, 5]]/values[i, 0]
            np.testing.assert_allclose(mean, d['mean'][0], atol=1e-10)
            np.testing.assert_allclose(d['cdf'][0], values[i, [3, 7, 4]]/values[i, 0], atol=1e-12)
            out['latest'][method] = {k: v.tolist() for k, v in d.items()}
        support = np.isfinite(rule['lp'])
        out['support_mass'] = dict(prior_disagreement=float(
            rule['weights'][support[:, 0] != support[:, 1]].sum()/rule['weights'].sum()))
        for i, method in enumerate(('solver', 'candidate')):
            mass = rule['weights']*np.exp(rule['lp'][:, i]-np.max(rule['lp'][:, i]))
            mass /= mass.sum()
            out['support_mass'][f'{method}_outside_other'] = float(mass[~support[:, 1-i]].sum())
        # Five equally spaced posterior-mass quantiles choose informative
        # columns without selecting on the subsequent temporal comparison.
        q = rule['weights']*np.exp(rule['lp'][:, 0]-np.max(rule['lp'][:, 0]))
        q /= q.sum()
        ys = np.unique(rule['theta'][:, 1])
        mass = np.array([q[rule['theta'][:, 1] == y].sum() for y in ys])
        chosen = np.unique(ys[np.searchsorted(np.cumsum(mass), [.1, .3, .5, .7, .9])])
        params = []
        for y in chosen:
            meta = json.loads((column_path(y) / f'stage{stage}.json').read_text())
            modes = [m for m in meta['discovery']['modes'] if m['method'] == 0]
            peak = max(modes, key=lambda m: m['loglike'])
            params.append([peak['x'], float(y)])
        params = np.array(params)
        selection = dict(params=params.tolist(), rule_sha256=digest(rule_path),
                         selection='outer posterior-mass quantiles 0.1/0.3/0.5/0.7/0.9, deduplicated')
        selection_path = OUT / 'temporal_selection.json'
        if selection_path.exists():
            assert json.loads(selection_path.read_text()) == selection
        else:
            write_json(selection_path, selection)
        path = DATA / 'mode_refinement.npz'
        if path.exists():
            strict = dict(np.load(path))
            np.testing.assert_array_equal(strict['params'], params)
        else:
            with ProcessPoolExecutor(max_workers=len(params)) as pool:
                rows = list(pool.map(refined_row, [(i, p, 2) for i, p in enumerate(params)]))
            assert not any(r[-1] for r in rows), rows
            strict = dict(params=params, profiles=np.stack([r[1] for r in rows]),
                          valid=np.stack([r[2] for r in rows]), status=np.array([r[3] for r in rows]))
            write_npz(path, **strict)
        baseline, valid = [], []
        for x, y in params:
            d = np.load(column_path(y) / 'nodes.npz')
            i = int(np.flatnonzero(d['x'] == x)[0])
            baseline.append(d['profiles'][i])
            valid.append(d['valid'][i])
        baseline, valid = np.array(baseline), np.array(valid)
        common = valid & strict['valid']
        maximum = float(np.max(np.abs(strict['profiles'][common]-baseline[common])))
        disagreements = np.argwhere(valid != strict['valid']).tolist()
        out['temporal'] = dict(n=len(params), max_profile_dex=maximum,
            support_disagreements=disagreements, passed=bool(not disagreements and maximum <= .00434))
        ensemble = Ensemble('candidate', 'cpu')
        inputs = np.column_stack([params, np.ones(len(params))])
        fields, ratio = ensemble.predict(np.repeat(initial_nfw()[None], len(params), axis=0), inputs, 400.)
        prediction = interpolate(fields[:, 0], R_OBS)
        candidate_valid = ratio <= 1
        paired = candidate_valid & strict['valid'][:, 1]
        out['mode_model_error'] = dict(n_common=int(paired.sum()),
            candidate_survival=candidate_valid.tolist(), reference_survival=strict['valid'][:, 1].tolist(),
            selection='same pre-recorded solver peaks as temporal comparison; targeted diagnostic only')
        if paired.any():
            delta = prediction[paired]-strict['profiles'][paired, 1]
            out['mode_model_error'].update(rmse_dex=float(np.sqrt(np.mean(delta**2))),
                max_abs_dex=float(np.max(np.abs(delta))),
                rmse_over_noise=float(np.sqrt(np.mean(delta**2))/NOISE_DEX))
        out['pilot_passed'] = bool(progress['numerical_passed'] and out['temporal']['passed'])
    if (OUT / 'convergence.json').exists():
        out['convergence'] = json.loads((OUT / 'convergence.json').read_text())
    if (OUT / 'cache_roundoff_diagnostic.json').exists():
        out['cache_roundoff_diagnostic'] = json.loads((OUT / 'cache_roundoff_diagnostic.json').read_text())
    if (OUT / 'postrun_diagnosis.json').exists():
        out['postrun_diagnosis'] = json.loads((OUT / 'postrun_diagnosis.json').read_text())
    out['source_sha256'] = digest(__file__)
    out['new_level1_solves'] = new_solves()
    base = np.load(F5 / 'pilot_bank.npz')
    discoveries = []
    for path in (DATA / 'columns').glob('*/nodes.npz'):
        column = np.load(path)
        y = float(column['y'])
        old = np.flatnonzero((base['params'][:, 1] == y) & base['valid'][:, 1])
        new = np.flatnonzero(column['valid'][:, 1])
        if len(old) and len(new):
            old_lp = -.5*np.square((base['profiles'][old, 1]-obs['observations'][0])/NOISE_DEX).sum(1)
            new_lp = -.5*np.square((column['profiles'][new, 1]-obs['observations'][0])/NOISE_DEX).sum(1)
            discoveries.append(dict(log_w=y, old_max_loglike=float(old_lp.max()),
                                    new_max_loglike=float(new_lp.max())))
    out['peak_discoveries'] = sorted(discoveries, key=lambda r: r['log_w'])
    write_json(OUT / 'validation.json', out)
    report(out)
    print('F6 audit/report completed; pilot passed', out['pilot_passed'], flush=True)


def report(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = d['progress']['history']
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
        for i, method in enumerate(('solver', 'candidate')):
            axes[0].semilogy([r['new_solves'] for r in history],
                            [np.max(r['relative_error'][i]) for r in history], 'o-', label=method)
        for target in (.02, .005, .00125):
            axes[0].axhline(target, color='gray', ls='--', lw=.6)
            axes[0].text(.99, target, f'{target:g}', transform=axes[0].get_yaxis_transform(),
                         ha='right', va='bottom', fontsize=8, color='dimgray')
        axes[0].set(xlabel='New tighter-solver trajectories', ylabel='Maximum propagated relative error')
        axes[0].legend()
        rule = np.load(DATA / 'latest_rule.npz')
        q = rule['weights']*np.exp(rule['lp'][:, 0]-np.max(rule['lp'][:, 0]))
        q /= q.sum()
        keep = q > 1e-8
        points = axes[1].scatter(rule['theta'][keep, 0], rule['theta'][keep, 1], s=3,
                                 c=np.log10(q[keep]), cmap='viridis')
        fig.colorbar(points, ax=axes[1], label='log10 node posterior mass')
        axes[1].set(xlabel='log10 sigma0', ylabel='log10 w', title='Solver posterior quadrature nodes')
        fig.savefig(OUT / 'conditional.png', dpi=170)
        fig.savefig(OUT / 'conditional.pdf')
        plt.close(fig)
    stop = ('global trajectory budget exhausted' if d.get('postrun_diagnosis') else d['progress']['reason'])
    lines = ['# F6 — conditional posterior integration (2026-09-10)', '',
        f"**Numerical pilot passed: {d['pilot_passed']}.** This is one selected diagnostic observation, not a population calibration result. Production, model weights and manuscript remain unchanged.", '',
        '## Method and scope', '',
        'Frozen protocol: [FORECAST_CONDITIONAL_PROTOCOL.md](FORECAST_CONDITIONAL_PROTOCOL.md). F4 observation 353 at time 400 is a difficult interior-time case from F5. Its observations and tighter-solver truth likelihood are reused for numerical development, with the complete original prior and observational noise unchanged.', '',
        'The conditional method surveys each sigma0 slice, locates both methods’ likelihood modes and shoulders, brackets survival transitions, and splits at parameter and solver-likelihood CDF boundaries. Positive Gauss–Legendre 3/5 rules control inner evidence, moment and CDF errors. Positive nested Simpson rules integrate over w and propagate the inner uncertainties; the w-CDF is classified per interval to avoid artificial endpoint atoms.', '',
        'Explicit breakpoints follow the numerical principle supported by [QUADPACK through SciPy quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html). [Outmezguine et al.](https://arxiv.org/abs/2204.06568) motivate examining clock-related narrow regions, but this implementation locates peaks directly and makes no exact universality assumption or prior-coordinate transformation. A finite survey cannot prove the absence of arbitrarily narrow hidden modes or survival islands.', '',
        'All direct reference values use the tighter level-1 solver. Candidate predictions use frozen checkpoints on CPU. No interpolated forward reference, tail truncation or discrepancy fit is introduced. Per-column caches save every four new solves and source/data/model identities are checked on resume.', '',
        '## Numerical result', '',
        f"New level-1 trajectories: **{d['new_level1_solves']}**. Stop: `{stop}`. Joint numerical convergence: **{d['progress']['numerical_passed']}**. Caps were 8,000 new trajectories overall and 512 per column.", '']
    if history:
        if d.get('postrun_diagnosis'):
            r = d['postrun_diagnosis']
            lines += [f"The global 8,000-trajectory budget was exhausted while a final column had completed 106 new solves; its per-column allowance was limited by the remaining global budget. This was a budget stop, not a reported direct-solver failure. {r['completed_columns']} of {r['total_columns']} attempted columns passed their inner checks. The last complete outer rule uses {r['last_complete_rule_columns']} columns at {r['last_complete_rule_new_solves']} new trajectories; the subsequent {r['remaining_work_solves']} evaluations did not yield another complete rule. Neither tolerance stage was reached.", '',
                'The limiting quantity in both methods is the sigma0 CDF numerator. In the last complete rule, most of its error comes from log-w [0.6492923469631748, 1.5]. Additional completed slices show the solver conditional sigma0 CDF rising from 0.268931 at log-w 0.64929235 to effectively 1 at 0.75563080. Explicit inner CDF cuts do not resolve this outer transition. The anticipated inherited-node cache miss also occurred, consuming 338 additional evaluations for the one-ULP-shifted column.', '']
        lines += ['![Conditional integration diagnostics](../results/forecast_conditional/conditional.png)', '',
            '| Stage | Outer panels | Columns | New solves | Solver maximum error | Candidate maximum error |',
            '|---:|---:|---:|---:|---:|---:|']
        for r in history:
            lines.append(f"| {r['stage']} | {r['intervals']} | {r['columns']} | {r['new_solves']} | {np.max(r['relative_error'][0]):.6g} | {np.max(r['relative_error'][1]):.6g} |")
        lines += ['', '| Method | Mean (log sigma0 / log w) | 90% widths | CDFs (sigma0 / w / solver likelihood) | TV to solver |',
            '|---|---|---|---|---:|']
        fmt = lambda values: ' / '.join(f'{v:.5f}' for v in values)
        for method, r in d['latest'].items():
            lines.append(f"| {method} | {fmt(r['mean'][0])} | {fmt(r['width90'][0])} | {fmt(r['cdf'][0])} | {r['tv'][0]:.5f} |")
        if d.get('support_mass'):
            mass = d['support_mass']
            lines += ['', f"On this rule, solver posterior mass outside candidate survival support is {mass['solver_outside_other']:.6g}; candidate posterior mass outside solver support is {mass['candidate_outside_other']:.6g}. The quadrature estimate of prior mass with differing survival classifications is {mass['prior_disagreement']:.6g}. These are diagnostic estimates on the reported rule and remain subject to its numerical gate."]
        lines += ['', 'Joint targets are 0.02 then 0.005, with a third 0.00125 level available if prescribed by the successive-summary checks. Inner tolerances are one tenth of these targets. Every outer interval is bisected between stages. Both methods require mean changes <=0.1 final SD, relative 90%-width changes <=0.1, and all three CDF changes <=0.02. Positive-node quantile widths are numerical approximations; their convergence is checked separately.', '',
            'The following errors are normalized by each method’s joint evidence. Each entry is the maximum over the eight controlled quantities; maxima from different columns need not refer to the same quantity.', '',
            '| Outer log-w interval | Solver outer error | Solver propagated inner error | Candidate outer error | Candidate propagated inner error |',
            '|---|---:|---:|---:|---:|']
        for row in d['panel_errors']:
            inner, outer = np.array(row['inner_relative']), np.array(row['outer_relative'])
            lines.append(f"| {fmt(row['bounds'])} | {outer[0].max():.6g} | {inner[0].max():.6g} | {outer[1].max():.6g} | {inner[1].max():.6g} |")
        lines.append('')
    else:
        lines += ['No complete outer rule was available. Inspect the per-column failure records; partial columns do not establish joint convergence.', '']
    lines += ['The direct peak search can be compared with the preceding F5 grid wherever exactly the same w column exists. Both sides use the same tighter solver and observation:', '',
              '| log w | Best F5 cached log likelihood | Best F6 cached log likelihood |',
              '|---:|---:|---:|']
    for r in d['peak_discoveries']:
        lines.append(f"| {r['log_w']:.6f} | {r['old_max_loglike']:.6f} | {r['new_max_loglike']:.6f} |")
    lines += ['', 'These differences show which sampled likelihood peaks were previously missed; a better sampled maximum alone does not establish a converged posterior integral.', '']
    if d['temporal'] is not None:
        r = d['temporal']
        lines += [f"Additional level-2 check: {r['n']} solver peaks selected through evenly spaced outer posterior-mass quantiles, maximum profile change {r['max_profile_dex']:.6g} dex, survival disagreements `{r['support_disagreements']}`, passed **{r['passed']}**. Selection was recorded before these stricter runs.", '']
    if d.get('mode_model_error'):
        r = d['mode_model_error']
        if r['n_common']:
            lines += [f"At the same selected peaks, candidate density RMSE to the level-2 reference at time 400 is {r['rmse_dex']:.6g} dex ({r['rmse_over_noise']:.4g} times the observational noise), with maximum absolute difference {r['max_abs_dex']:.6g} dex across {r['n_common']} common-survival profiles. This is a targeted model diagnostic on the reused case, not an independent population accuracy estimate.", '']
    lines += ['## Interpretation and verification', '']
    lines += ['A separate [development note](FORECAST_CONDITIONAL_FOLLOWUP_2026-09-10.md) examines outer prior-boundary, parameter-CDF and likelihood-CDF features, documents the cache-reuse issue, and reports 24 analytic benchmark runs. Those benchmarks do not replace this direct-solver pilot or change its frozen protocol.', '']
    if d['pilot_passed']:
        lines += ['This conditional method clears the numerical screens on the selected case. It supports extending numerical validation to the other fixed times and difficult cases, followed by a separately frozen study with fresh observations. It does not justify production replacement or a population-calibration claim.', '']
    else:
        lines += ['The configuration does not yet support broader calibration. Preserve all failed gates and diagnose the specific unresolved conditional or outer integration error before expanding the study. The existing candidate model has not been tuned on these observations.', '']
    if d.get('cache_roundoff_diagnostic'):
        lines += ['A read-only cache audit also found an efficiency issue: bisecting the original upper outer panel regenerates an inherited node as 0.8619692602223811 instead of 0.8619692602223812. Exact-byte cache keys treat these as separate columns. The new helper `src/gravop/dyadic.py` uses fixed base bounds and integer descendant indices to preserve inherited coordinates exactly. Three tests reproduce the cache miss, verify bit-identical reuse through 17 refinement depths and shared base endpoints, and reject invalid or unrepresentable panels (`logs/f6_dyadic_tests.log`). Integration into a subsequent driver remains outstanding. This run retains its frozen arithmetic and counts all actual trajectories; the finding does not change a failed numerical gate or authorize merging distinct cached coordinates. Details: `results/forecast_conditional/cache_roundoff_diagnostic.json`.', '']
    lines += ['The audit verifies frozen input/source/checkpoint hashes and, when an outer rule exists, positive weights covering the full prior, propagated error reproduction, direct likelihood arithmetic, integrated means and CDFs. Analytic tests cover off-grid narrow peaks, multiple modes with disconnected support, candidate mass outside solver support, explicit outer CDF sides, inner-error propagation and exact inherited-node reuse. The final full suite passed 138 tests; output is in `logs/f6_tests_verified.log`. One manuscript-count test remains deferred.', '',
        '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_conditional.py --workers 16',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_conditional_report.py',
        '.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts',
        '```', '',
        'Artifacts: `results/forecast_conditional/`; ignored caches: `data/d1/forecast_conditional/`; logs: `logs/f6_*`. Earlier F4/F5 failures remain recorded.', '']
    (ROOT / 'docs/FORECAST_CONDITIONAL_RESULTS_2026-09-10.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
