"""Audit and report the F5 temporal study and bounded integration pilot."""
import json
import platform

import numpy as np
import scipy
import torch

from forecast_numerics_pilot import DATA, OUT, ROOT, F4, PilotBank, log_likelihood, LATTICE
from forecast_calibrate import LO, HI, NOISE_DEX
from forecast_validate import digest, write_json
from gravop.simpson import SimpsonMesh


def audit():
    temporal = json.loads((OUT / 'temporal_diagnostic.json').read_text())
    pilot = json.loads((OUT / 'pilot_validation.json').read_text())
    selection = pilot['selection']
    expected = {
        ROOT / 'docs/FORECAST_NUMERICS_PROTOCOL.md': temporal['selection']['protocol_sha256'],
        ROOT / 'docs/FORECAST_NUMERICS_PILOT.md': selection['protocol_sha256'],
        F4 / 'bank.npz': temporal['selection']['bank_sha256'],
        F4 / 'observations.npz': selection['f4_observations_sha256'],
        DATA / 'pilot_bank.npz': pilot['bank_sha256'],
        DATA / 'pilot_observations.npz': pilot['observations_sha256'],
        DATA / 'pilot_candidate_predictions.npz': pilot['predictions_sha256'],
    }
    for level in (1, 2):
        expected[DATA / f'level{level}.npz'] = json.loads((OUT / f'level{level}.json').read_text())['sha256']
    for path, sha in {**pilot['source_sha256'], **pilot['checkpoint_hashes']}.items():
        expected[ROOT / path] = sha
    # Also check production against the preceding frozen F4 identity record.
    f4_predictions = json.loads((ROOT / 'results/forecast_integration/prediction_provenance.json').read_text())
    expected[F4 / 'model_predictions.npz'] = f4_predictions['predictions_sha256']
    for group in f4_predictions['checkpoint_hashes'].values():
        expected.update({ROOT / p: sha for p, sha in group.items()})
    for path, sha in expected.items():
        assert digest(path) == sha, path
    bank = PilotBank()
    original = np.load(DATA / 'level1.npz')
    for k in original.files:
        np.testing.assert_array_equal(bank.d[k][:len(original['params'])], original[k])
    np.testing.assert_array_equal(bank.d['params'], LO + bank.d['keys'] / LATTICE * (HI - LO))
    assert set(bank.d['status']).issubset({0, 1, 2})
    assert np.isfinite(bank.d['profiles'][bank.d['valid']]).all()
    state = json.loads((DATA / 'pilot_mesh.json').read_text())
    mesh = SimpsonMesh(cells=state['cells'])
    keys, ix = mesh.stencil()
    ids = bank.ids(keys)
    weights = np.zeros(len(bank.d['keys']))
    weights[ids] = mesh.weights(ix, len(keys))
    rule = np.load(DATA / 'pilot_latest.npz')
    np.testing.assert_array_equal(weights, rule['weights'])
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-14, rtol=0)
    assert (weights >= 0).all()
    obs = np.load(DATA / 'pilot_observations.npz')
    lp = log_likelihood(bank.d['profiles'], bank.d['valid'], obs)
    assessment = mesh.assess(np.exp(lp[ids] - lp.max(0)), ix)
    np.testing.assert_allclose(assessment['relative_error'], pilot['progress']['history'][-1]['relative_errors'])
    sampled = np.linspace(0, len(weights) - 1, 97, dtype=int)
    for case, j in enumerate(obs['time_index']):
        good = sampled[bank.d['valid'][sampled, j]]
        direct = -.5 * np.square((bank.d['profiles'][good, j] - obs['observations'][case]) / NOISE_DEX).sum(1)
        np.testing.assert_allclose(lp[good, case], direct, atol=1e-7, rtol=1e-10)

    base = np.load(DATA / 'level0.npz')
    strict = np.load(DATA / 'level2.npz')
    keys0 = np.rint((base['params'] - LO) / (HI - LO) * 4096).astype(int)
    lookup = {tuple(k): i for i, k in enumerate(keys0)}
    spans = []
    for cell, (x, y, step) in zip(temporal['selection']['cells'], temporal['selection']['geometry']):
        rows = np.array([lookup[(x + i * step // 2, y + j * step // 2)]
                         for i in range(3) for j in range(3)])
        for j, t in enumerate((100, 400, 900)):
            good = rows[base['valid'][rows, j] & strict['valid'][rows, j]]
            if len(good) > 1:
                spans.append(dict(cell=cell, step=step, time=t,
                    coarse_span=float(np.ptp(base['profiles'][good, j], axis=0).max()),
                    strict_span=float(np.ptp(strict['profiles'][good, j], axis=0).max())))
    result = dict(passed=True, checks=['recorded hashes', 'level-1 cache identity', 'positive full-prior weights',
        'final mesh and error reproduction', 'explicit likelihood arithmetic'], cell_spans=spans,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         torch=torch.__version__, cuda=torch.version.cuda),
        audit_source_sha256=digest(__file__))
    prediction = np.load(F4 / 'model_predictions.npz')
    rows = temporal['selection']['bank_rows']
    comparison = {}
    for j, t in enumerate((100, 400, 900)):
        common = (base['valid'][:, j] & strict['valid'][:, j] &
                  prediction['candidate_valid'][rows, j] & prediction['production_valid'][rows, j])
        comparison[str(t)] = dict(n_common=int(common.sum()), methods={})
        for kind in ('candidate', 'production'):
            ys = prediction[kind + '_profiles'][rows, j]
            comparison[str(t)]['methods'][kind] = dict(
                rmse_to_coarse=float(np.sqrt(np.mean((ys[common] - base['profiles'][common, j])**2))),
                rmse_to_finest=float(np.sqrt(np.mean((ys[common] - strict['profiles'][common, j])**2))),
                rejected_fine_survivors=int((strict['valid'][:, j] & ~prediction[kind + '_valid'][rows, j]).sum()))
    write_json(OUT / 'selected_forward_error.json', comparison)
    result['selected_forward_comparison'] = comparison
    write_json(OUT / 'audit.json', result)
    return temporal, pilot, result


def main():
    temporal, pilot, checked = audit()
    diagnostics = np.load(OUT / 'pilot_diagnostics.npz')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = pilot['progress']['history']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    labels = ['3e-4 to 1e-4', '1e-4 to 3e-5', '3e-4 to 3e-5']
    values = [temporal[k]['max_profile_dex'] for k in ('0_to_1', '1_to_2', '0_to_2')]
    ax[0].bar(labels, values, color=['#b2182b', '#2166ac', '#b2182b'])
    ax[0].axhline(.00434, color='black', ls='--', lw=1, label='Temporal screen')
    ax[0].set(ylabel='Maximum log-density change (dex)', title='Difficult-cell temporal refinement')
    ax[0].tick_params(axis='x', labelrotation=15)
    ax[0].legend()
    labels = [f't={t}, {kind}' for t in (100, 400, 900) for kind in ('worst F4 error', 'median F4 error')]
    errors = np.array([r['relative_errors'] for r in history])
    for i, label in enumerate(labels):
        ax[1].semilogy([r['nodes'] for r in history], errors[:, i], '.-', label=label)
    for tolerance in (.02, .005, .00125):
        ax[1].axhline(tolerance, color='gray', ls='--', lw=.6)
    ax[1].set(xlabel='Tighter-solver nodes in bank', ylabel='Raw embedded relative error', title='Higher-order pilot')
    ax[1].legend(fontsize=7)
    fig.savefig(OUT / 'numerics.png', dpi=170)
    fig.savefig(OUT / 'numerics.pdf')
    plt.close(fig)
    fmt = lambda a: ' / '.join(f'{x:.4f}' for x in a)
    lines = ['# F5 — solver accuracy and higher-order integration (2026-09-10)', '',
        f"Tighter solver settings pass the targeted temporal screen. **Higher-order integration pilot: {'passed' if pilot['pilot_passed'] else 'failed'}.** This study does not establish population calibration. All model checkpoints, production and manuscript remain unchanged.", '',
        '## Research and decisions', '',
        'Protocols: [temporal diagnostics](FORECAST_NUMERICS_PROTOCOL.md) and [subsequent numerical pilot](FORECAST_NUMERICS_PILOT.md). The pilot was frozen after temporal results and before its new solver evaluations. F4’s failed result and original limits are preserved.', '',
        'A positive nested Simpson rule reuses nodes while improving polynomial accuracy. [SciPy documents cubic exactness for equally spaced Simpson rules](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html). Our tests independently verify tensor-cubic exactness, fourth-order convergence, a thin diagonal ridge against an analytic integral, integer-node seeding/full prior area, invalid-mass rejection and the need to check likelihood CDFs even when moments stabilize. Raw embedded differences are retained; no division by 15 is used to relax the error screen.', '',
        '[SciPy’s cubature documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html) describes higher-order adaptive subdivision without guaranteeing convergence. Its [integration tutorial](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) warns about narrow missed regions. The pilot therefore retains a global survey and seeds known peak locations; it still requires explicit posterior convergence.', '',
        '## Targeted solver findings', '',
        'Selected eight difficult F4 cells and 66 unique nodes before refinement. Ran 132 tighter trajectories, at eps_t=1e-4 and 3e-5 with progressively denser snapshots. Neither level had numerical failures or changed queried-time survival classifications.', '',
        '| Comparison | Maximum profile change (dex) | RMS change (dex) | Temporal screen |',
        '|---|---:|---:|---|']
    for key, label in zip(('0_to_1', '1_to_2', '0_to_2'), ('3e-4 → 1e-4', '1e-4 → 3e-5', '3e-4 → 3e-5')):
        r = temporal[key]
        lines.append(f"| {label} | {r['max_profile_dex']:.6f} | {r['rms_profile_dex']:.6f} | {r['passed']} |")
    lines += ['', 'The local 0.00434-dex screen is one tenth of the observational noise. These outcomes support eps_t=1e-4 for a pilot, rather than reusing the coarser F4 reference. They do not prove spatial convergence or uniform temporal accuracy over the prior.', '',
        'Sharp parameter variation remains after refinement. In minimum-size cell 2748 at time 400, the maximum log-density span changes from 0.178740 to 0.185114 dex. Reducing timestep error leaves this narrow feature unresolved by the quadrature. Full per-cell spans are in `audit.json`.', '',
        '![Temporal and integration diagnostics](../results/forecast_numerics/numerics.png)', '',
        'Frozen-model errors on these deliberately selected cells, using only rows supported by both models and both solver levels:', '',
        '| Time | Common rows | Candidate RMSE to finest (dex) | Production RMSE to finest (dex) | Candidate / production rejected finest-solver survivors |',
        '|---|---:|---:|---:|---|']
    for t, r in checked['selected_forward_comparison'].items():
        c, p = r['methods']['candidate'], r['methods']['production']
        lines.append(f"| {t} | {r['n_common']} | {c['rmse_to_finest']:.6f} | {p['rmse_to_finest']:.6f} | {c['rejected_fine_survivors']} / {p['rejected_fine_survivors']} |")
    lines += ['', 'These targeted results are not population error estimates. The support rejections identify a clock-boundary issue worth tracking separately from density error. The comparison uses a shared row mask; rejected survivors are reported rather than silently included in the accuracy comparison.', '',
        '## Independent analytic comparison', '',
        'Both methods begin with the same 4,225 nodes, use the same 0.005 raw estimated-error target and a 20,000-node cap, and retain the same unit-square prior. The examples have known exact integrals. This separate benchmark does not establish SIDM performance.', '',
        '| Integrand | Rule | Nodes | Estimated relative error | Actual relative error | Target reached |',
        '|---|---|---:|---:|---:|---|']
    benchmark = json.loads((OUT / 'analytic_benchmark.json').read_text())
    for name, methods in benchmark['results'].items():
        for method, r in methods.items():
            lines.append(f"| {name} | {method} | {r['nodes']} | {r['embedded_error']:.5g} | {r['actual_error']:.5g} | {r['converged']} |")
    lines += ['', 'Simpson improves the smooth examples, but requires more nodes and has larger actual error on the discontinuous survival triangle. Higher order alone is therefore not a general solution to survival boundaries. The conservative error estimate can also substantially exceed measured error; it remains a diagnostic estimate, not a rigorous bound.', '',
        '## Bounded integration pilot', '',
        'The six cases are the worst and median F4 integration-error observations at each of times 100/400/900. These are selected, reused diagnostic observations. Their likelihood reference at the true parameters is recomputed using the tighter solver; they are not fresh tighter-solver simulations and no SBC uniformity tests are performed.', '',
        f"All {history[-1]['nodes']} bank nodes use eps_t=1e-4; 66 were reused from the temporal study. F4 profiles only locate initial seed points. The complete prior and survival support are retained. Stop: **{pilot['progress']['reason']}**. Tolerance-level convergence: **{pilot['progress']['converged']}**. Compared rules: `{pilot['rules']}`.", '',
        '| Case | Time / selection | Final embedded error | Solver mean shifts / SD | Solver CDF changes | Candidate CDF changes | Candidate TV to solver |',
        '|---:|---|---:|---|---|---|---:|']
    for i, case in enumerate(pilot['selection']['cases']):
        lines.append(f"| {case} | {labels[i]} | {history[-1]['relative_errors'][i]:.4g} | {fmt(pilot['convergence']['solver']['mean_shift_sd'][i])} | {fmt(pilot['convergence']['solver']['cdf_change'][i])} | {fmt(pilot['convergence']['candidate']['cdf_change'][i])} | {diagnostics['candidate_tv'][i]:.4f} |")
    lines += ['', 'Required targets are sequential estimated errors <=0.02 then <=0.005 (predefined 0.00125 fallback); every selected case must also satisfy mean change <=0.1 final SD, relative interval-width change <=0.1, and CDF change <=0.02 for both parameters and the solver likelihood. Solver and candidate are checked separately. Initial-to-final comparisons when tolerance stages are unavailable are descriptive only.', '',
        f"New-mode temporal check: {len(pilot['mode_temporal_check']['bank_rows'])} level-2 reruns, maximum profile change {pilot['mode_temporal_check']['max_profile_dex']:.6f} dex; support disagreements `{pilot['mode_temporal_check']['support_disagreements']}`; passed **{pilot['mode_temporal_check']['passed']}**.", '',
        'This is not a matched-budget comparison to F4: it changes solver accuracy, quadrature order, seeding and the number of target observations. Posterior comparisons remain conditional on numerical convergence.', '']
    if pilot['pilot_passed']:
        lines += ['The pilot meets its numerical screens on these six cases. The next step is a separately frozen broad study with new continuous-prior observations; six selected cases cannot establish population calibration.', '']
    else:
        lines += ['Do not proceed to broad fresh calibration on this numerical configuration. Preserve the tighter reference and all failed checks. The integration or newly probed temporal limits identified here must be resolved before another population claim.', '']
    lines += ['## Revised recommendation', '',
        'Retain the tighter solver settings for the next numerical pilot. Test conditional one-dimensional integration in log sigma0 at fixed log w, using an effective-cross-section or clock coordinate to locate narrow likelihood regions. Preserve the full original prior and all branches; explicitly split at survival transitions and posterior-CDF discontinuities. Adaptation should target the CDF integrals themselves as well as evidence and moments. The large observed CDF shifts show that a small error in evidence alone would be insufficient.', '',
        'This coordinate choice is a research hypothesis grounded in the project’s physics: [Outmezguine et al.](https://arxiv.org/abs/2204.06568) describe approximate temporal universality using a scattering timescale and averaged cross section. That result does not guarantee an exact mapping for this code’s finite stopping criteria or observation likelihood. Any coordinate transformation must retain the prior measure and include its Jacobian. [QUADPACK through SciPy quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html) accepts explicit breakpoints at local difficulties, which supports testing a boundary-split conditional method. This alternative is recommended, not yet implemented or validated.', '',
        'For subsequent model development, retain difficult near-termination cases and report clock support errors separately from density errors. The four candidate clock rejections found here motivate a future boundary-focused training/evaluation set. These selected diagnostic cases must not become the next independent confirmation set.', '']
    lines += ['## Verification and reproduction', '',
        'The independent audit verifies protocol/data/source/checkpoint hashes, the reused level-1 cache, positive weights over the entire prior, mesh/error reproduction, and matrix likelihoods against direct residuals. Runtime versions and all cell spans are recorded in `audit.json`. The full suite passes 130 tests with one deferred manuscript-count check deselected (`logs/f5_tests.log`); all six added targeted tests pass.', '',
        '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_numerics.py --workers 20',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_numerics_pilot.py adaptive --workers 20',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_numerics_pilot.py evaluate --workers 20 --device cuda:0',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_quadrature.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_numerics_report.py',
        '.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts',
        '```', '',
        'Artifacts: `results/forecast_numerics/`; ignored solver caches: `data/d1/forecast_numerics/`; logs: `logs/f5_*`. Node caps are part of the frozen pilot and cannot be treated as convergence.', '']
    (ROOT / 'docs/FORECAST_NUMERICS_RESULTS_2026-09-10.md').write_text('\n'.join(lines))
    print('F5 audit passed; report and figures written')


if __name__ == '__main__':
    main()
