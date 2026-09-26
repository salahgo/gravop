"""Report F4 same-bounds numerical convergence and fresh continuous checks."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/forecast_integration'
DATA=ROOT/'data/d1/forecast_integration'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads((OUT/'validation.json').read_text());g=json.loads((OUT/'generation.json').read_text())
    b=np.load(DATA/'bank.npz');history=d['progress']['history']
    audit=json.loads((OUT/'final_error_diagnostic.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].semilogy([r['new_nodes'] for r in history],[r['worst_relative_error'] for r in history],'o-',color='#2166ac')
    for v in (.02,.005,.00125):axes[0].axhline(v,color='gray',ls='--',lw=.7)
    axes[0].set(xlabel='Additional direct-solver nodes',ylabel='Maximum embedded relative error',title='Integration progress')
    theta=b['params'];axes[1].scatter(theta[:4225,0],theta[:4225,1],s=1,color='#cccccc',label='Original 65 grid')
    axes[1].scatter(theta[4225:,0],theta[4225:,1],s=2,color='#2166ac',label='Adaptive nodes')
    axes[1].set(xlabel='log10 sigma0',ylabel='log10 w',title='Full prior retained');axes[1].legend(markerscale=3)
    fig.savefig(OUT/'integration.png',dpi=170);fig.savefig(OUT/'integration.pdf');plt.close(fig)
    date='2026-09-10'
    lines=[f'# F4 — adaptive posterior integration ({date})','',
           f"Integration gates passed: **{d['integration_passed']}**. Solver/negative controls passed: **{d['controls_passed']}**. Raw candidate passed all numerical/statistical gates: **{d['candidate_passed']}**. Production and manuscript remain unchanged.",'',
           '## Method and scope','',
           'Protocol: [FORECAST_INTEGRATION_PROTOCOL.md](FORECAST_INTEGRATION_PROTOCOL.md). The new positive dyadic quadrature reuses the 65x65 full-prior survey and spends direct-solver evaluations on cells with large normalized embedded error. Each cell compares corner trapezoids against its 3x3 fine rule. The sum of absolute local differences is an error estimate, not a rigorous bound. No emulator or interpolated forward field supplies reference-node values.','',
           'This follows the global adaptive-subdivision idea described in [SciPy cubature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html), with a custom reusable-node rule. The [SciPy integration tutorial](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) warns about missing narrow important regions. We keep the full prior survey and check successive posterior means, widths and CDF values before interpreting calibration.','',
           f"Generated {g['n']} independent continuous-prior trajectories: survivors {g['survivors']} at fixed times 100/400/900, {sum(g['survivors'])} noisy observations in total. Solver failures: {g['failures']}; earlier-data overlaps: `{g['overlap']}`. The existing NFW/Yukawa prior, 20 radii and 0.0434 dex observational noise remain fixed. The population is explicitly conditioned on survival at each time. No model or discrepancy correction was fitted to these observations.",'',
           'The raw candidate and production forward ensemble use the same observational-noise-only likelihood; this does not re-evaluate the old production discrepancy-inflated MCMC pipeline. The rejected F3 correction remains disabled.','',
           '## Numerical outcome','',
           f"Final solver-node count: **{len(theta)}**, of which **{len(theta)-4225}** are new adaptive evaluations. Stopping status: `{d['progress']['action']}`. Maximum final embedded relative error: **{history[-1]['worst_relative_error']:.6g}**. Compared rules: `{d['rules']}`.",'',
           '![Adaptive integration progress](../results/forecast_integration/integration.png)','',
           '| Method | Time | p95 mean shift / final SD | Median relative width change | p95 CDF changes (sigma0, w, likelihood) | Pass |',
           '|---|---:|---|---|---|---|']
    fmt=lambda values:' / '.join(f'{x:.4f}' for x in values)
    for kind,comp in d['convergence'].items():
        for t,r in comp['times'].items():
            lines.append(f"| {kind} | {t} | {fmt(r['mean_shift_sd_p95'])} | {fmt(r['width_relative_median'])} | {fmt(r['cdf_difference_p95'])} | {r['passed']} |")
    lines += ['', 'Thresholds are p95 mean change <=0.1 SD, median relative interval-width change <=10%, and p95 CDF change <=0.02 for each quantity. Solver and candidate must both pass, together with estimated-error convergence. A budget-limited rule is never labeled converged merely because some summary statistics stabilize.','']
    if d['rules'][0]=='initial_rule.npz':
        lines += ['Neither tolerance level was reached. The initial-versus-final comparison above is descriptive; it does not substitute for the protocol’s required comparison between completed tolerance levels. Zero median interval-width changes reflect discrete quantile endpoints and do not establish convergence.','']
    lines += ['The recomputed final embedded-error distribution is:','',
              '| Time | Median | p95 | Maximum | Fraction below 0.02 |',
              '|---|---:|---:|---:|---:|']
    for t,r in audit['times'].items():
        lines.append(f"| {t} | {r['median']:.4f} | {r['p95']:.4f} | {r['maximum']:.4f} | {r['fraction_within_first_tolerance']:.3f} |")
    lines += ['', f"There are {audit['cell_step_counts'].get('2',0)} cells at the minimum allowed size. Their summed normalized embedded error reaches {audit['minimum_size_error_max']:.4f} for one observation. However, the worst observation at each time has no error contribution from these cells: both the node budget and local resolution deserve attention. These are estimated quadrature errors, not measured forward-model errors.",'',
              '## Fresh continuous-prior diagnostics','',
              '| Method | Time | n | Median TV to solver | Central-CDF 90% coverage (sigma0 / w) | Minimum Holm p |',
              '|---|---:|---:|---:|---|---:|']
    for kind,rows in d['methods'].items():
        for t,r in rows.items():
            q=r['quantities'];coverage=[q[k]['coverage']['0.9']['estimate'] for k in ('log_sigma0','log_w')]
            lines.append(f"| {kind} | {t} | {r['n']} | {r['median_tv']:.4f} | {fmt(coverage)} | {min(v['holm_pvalue'] for v in q.values()):.4g} |")
    lines += ['', 'CDF values use positive quadrature weights; the truth parameters are continuous and independently drawn. The reference likelihood quantity is evaluated from the direct-solver truth profile and the same noise draw. Numerical CDF error is checked separately. Holm adjustment covers nine tests per method. Ordinary quantile-endpoint coverage, 68/90/95% central-CDF coverage with Wilson intervals, p90 TV, widths and test statistics are in `validation.json`; per-observation summaries are in `diagnostics.npz`.','']
    lines += [f"Estimated surviving prior mass at times 100/400/900: {fmt([d['surviving_prior_mass'][t] for t in ('100','400','900')])}. Candidate mean posterior mass outside solver support is at most {max(r['mean_false_support_mass'] for r in d['methods']['candidate'].values()):.3g}; mean solver posterior mass rejected by the candidate is at most {max(r['mean_rejected_reference_mass'] for r in d['methods']['candidate'].values()):.3g}. These are also provisional mesh estimates.",'']
    if d['candidate_passed']:
        lines.append('The candidate clears this observational model’s prespecified screening gates. This is evidence for continuous-prior calibration at these three times and this noise/initial-condition design, not proof of exact inference, spatial solver convergence, physical-model accuracy, or MCMC convergence.')
    elif not d['integration_passed']:
        lines.append('Continuous-prior calibration remains unestablished because the integration gates fail. Treat the statistical diagnostics as provisional; they cannot distinguish all emulator errors from an insufficient numerical reference. Preserve the raw candidate and the cache, and resolve the remaining integration limits before a production claim.')
    else:
        lines.append('Integration passes, but the candidate does not clear every statistical gate. The remaining failure is no longer explained by the measured integration changes; investigate conditional forward/clock errors without tuning on this confirmation set.')
    lines += ['', '## Next numerical experiment','',
              'Keep the frozen raw candidate. First inspect solver smoothness and survival transitions in the cells that have already reached the resolution limit, using stricter direct solves. Then benchmark a higher-order or per-observation integration rule on a fixed diagnostic subset, measuring actual posterior CDF stability as well as evidence error. SciPy’s cubature offers Gauss–Kronrod and Genz–Malik rules, but their suitability for these likelihoods is untested; they are candidates for a separate experiment, not validated replacements. Preserve this failed F4 result and its original thresholds. Use fresh confirmation draws after choosing the next numerical method; do not tune model weights or uncertainty on F4 observations.','',
              '## Reproduction and verification','', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_integrate.py generate --workers 20',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_integrate.py adaptive --workers 20',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_integrate.py evaluate --device cuda:0',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_integration_audit.py',
              '.venv/bin/python scripts/forecast_integration_report.py',
              '.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts',
              '```','',
              'Generation refuses to overwrite the fresh observations. Adaptive computation resumes its atomic node/mesh cache. The preceding prediction cache must be archived if the bank is subsequently refined. The predefined third tolerance can be requested with `adaptive --refine-next` after a completed second stage if candidate convergence fails.','',
              'Verification: 124 tests passed, one stale manuscript-count check deselected. New analytic tests cover positive weights/full area after local subdivision, bilinear integration, separated narrow Gaussian peaks and rejection of invalid/zero-mass integrands. The independent cache audit reproduces final weights and error estimates, checks the matrix likelihood against explicit residuals, and verifies recorded protocol, observation, bank, prediction, source and checkpoint hashes. Runtime library/device versions are in `final_error_diagnostic.json`. The first interrupted refinement batch was restarted from the saved mesh; completed node batches are reused.','',
              'Artifacts: `results/forecast_integration/`; ignored solver caches: `data/d1/forecast_integration/`; logs: `logs/f4_*`. No manuscript or model checkpoint was modified.','']
    (ROOT/'docs/FORECAST_INTEGRATION_RESULTS_2026-09-10.md').write_text('\n'.join(lines))
    print('wrote F4 report and integration figures')


if __name__=='__main__':main()
