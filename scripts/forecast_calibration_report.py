"""Render the completed F3 records without altering selections or checkpoints."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/forecast_calibration'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads((OUT/'validation.json').read_text())
    cal=json.loads((OUT/'discrepancy.json').read_text())
    diag=np.load(OUT/'diagnostics.npz')
    times=list(d['times']); quantities=('log_sigma0','log_w','solver_log_likelihood')
    labels=('log10 sigma0','log10 w','Solver log likelihood')
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True,sharey='col',layout='constrained')
    colors={'solver':'#666666','candidate_raw':'#2166ac','candidate_corrected':'#1b7837',
            'production_corrected':'#b2182b','prior_only':'#ae8800'}
    for i,t in enumerate(times):
        for j,label in enumerate(labels):
            ax=axes[i,j];n=d['times'][t]['n'];band=np.sqrt(np.log(2/.05)/(2*n))
            ax.axhspan(-band,band,color='#eeeeee',label='95% DKW band per curve')
            for kind in colors:
                if kind=='prior_only' and j!=2:continue
                v=np.sort(diag[f'{t}_{kind}_pit'][:,j])
                x=np.r_[0.,v,1.];y=np.r_[0.,np.arange(1,len(v)+1)/len(v),1.]-x
                ax.plot(x,y,color=colors[kind],lw=1.2,label=kind.replace('_',' '))
            ax.axhline(0,color='black',lw=.5);ax.set_title(f't = {t}: {label}')
            ax.set_xlim(0,1);ax.set_ylim((-.8,.15) if j==2 else (-.15,.15))
            if i==2:ax.set_xlabel('Randomized posterior CDF')
            if j==0:ax.set_ylabel('Empirical CDF minus uniform CDF')
    handles,legend=axes[0,2].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('F3 calibration diagnostics: exact finite-grid prior, solver-generated observations')
    fig.savefig(OUT/'calibration_ecdf.png',dpi=160);fig.savefig(OUT/'calibration_ecdf.pdf');plt.close(fig)
    lines=['# F3 — inference fixes and population calibration','',
           '2026-09-09. User-directed research, fixes and next validation stage. Manuscript deferred.','',
           '## Outcome','',
           'Retain the raw candidate and reject the selected global discrepancy correction. The raw candidate passes all nine finite-grid uniformity checks and the prespecified coverage/median-TV screens. The correction improves agreement at time 100 but worsens it at 900, where the data-dependent likelihood test rejects calibration (Holm p=0.0151). Parameter-only tests would have missed this failure.','',
           f"Overall advancement gates: `{d['candidate_can_advance']}`. Positive/negative controls pass: **{d['controls_passed']}**. Numerical solver spot checks pass, but all three same-bounds grid-resolution checks fail. Continuous-prior calibration is therefore not established. Production remains unchanged; passing a rank test is not proof of exact calibration.",'',
           '## What changed and why','',
           'The standalone forecast loader now validates finite, positive normalization scales and catches nonfinite/overflowed predicted terminal times before returning horizon flags. It exposes the predicted terminal time, preserves exact initial fields, and checks finite outputs. Model weights and the learned field equation are unchanged.','',
           'Added `gravop.calibration`: stable posterior normalization with explicit rejection of empty support; a Cholesky Gaussian likelihood with mean bias, correlated covariance and its normalization term; independently selected shrinkage corrections; exact randomized posterior CDFs with tied values; coverage intervals and Holm multiple-testing adjustment. Data writes are atomic, resumed prediction caches verify their data/checkpoint hashes, and calibration/confirmation identities are audited.','',
           'The method follows [Talts et al.](https://arxiv.org/abs/1804.06788) and the [Stan SBC guide](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html). [Modrak et al.](https://arxiv.org/abs/2211.02383) motivate the data-dependent likelihood test: parameter ranks alone can miss a posterior that ignores data. [Sailynoja et al.](https://arxiv.org/abs/2103.10522) motivate uniformity diagnostics; [Hermans et al.](https://arxiv.org/abs/2110.06581) motivate checking coverage and informativeness together. [Covariance shrinkage](https://scikit-learn.org/stable/modules/covariance.html) motivates regularization; our candidate correction menu and acceptance tolerances are project choices.','',
           '## Frozen experiment','',
           'Protocol: [FORECAST_CALIBRATION_PROTOCOL.md](FORECAST_CALIBRATION_PROTOCOL.md). NFW initial conditions, Yukawa cross sections, the existing uniform log-parameter prior, 20 observed density radii and 0.0434 dex Gaussian noise. Times 100, 400 and 900 are fixed physical times, not truth-dependent fractions of terminal time. The generating prior is explicitly conditioned on the solver surviving to the observation time.','',
           'A 65x65 grid with trapezoidal prior weights supplies an exact finite-prior reference. Each time has 512 independent prior/noise simulations, 1,536 total. Both methods see the same observations. Their raw and corrected likelihoods use the same observation noise. The production comparison uses its existing forward ensemble in this common likelihood framework, not its older discrepancy-inflated MCMC likelihood.','',
           'Corrections use 256 independent continuous-prior trajectories: first 192 for fitting, remaining 64 for selection, retaining survivors at each time. A further 128 untouched continuous truths check coverage and the nested 33/65 grids on the same prior bounds. Twelve prespecified direct-solver refinements check numerical error. No test result selects a correction or retunes a neural model.','',
           '| Time | Surviving grid-prior mass | Calibration fit/selection | Continuous confirmations |','|---|---:|---:|---:|']
    for t in times:
        c=cal['candidate'][t]
        lines.append(f"| {t} | {d['times'][t]['solver_surviving_prior_mass']:.3f} | {c['n_fit']}/{c['n_selection']} | {d['continuous'][t]['n']} |")
    lines+=['','Generation numerical failures: `'+str(d['generation_failures'])+'`. Calibration and confirmation share no parameter identities with each other or the old Yukawa D1 data at float32 input precision.','',
            '| Model | Time | Selected correction | Selected minus raw held-out log score |','|---|---:|---|---:|']
    for kind in ('candidate','production'):
        for t in times:
            c=cal[kind][t];scores=c['selection_log_scores']
            lines.append(f"| {kind} | {t} | {c['selected']} | {scores[c['selected']]-scores['none']:.4f} |")
    lines+=['','The corrected mean adds the fitted solver-minus-emulator bias. Covariance is fitted to clean residuals, then observational variance is added. Shrinkage values 0.1/0.5/1 blend the full covariance toward its diagonal. Selection uses shared noisy validation residuals across options. The correction is frozen without refitting; uncertainty in the fitted correction is not itself marginalized.','',
            '## Exact finite-grid SBC','',
            '| Method | Time | Median TV to solver | 90% PIT coverage, sigma0 / w | Minimum Holm p (three quantities at this time) |','|---|---:|---:|---|---:|']
    for kind in ('solver','candidate_raw','candidate_corrected','production_raw','production_corrected','prior_only'):
        for t in times:
            r=d['times'][t]['methods'][kind];q=r['test_quantities'];cov=[q[k]['coverage']['0.9']['estimate'] for k in quantities[:2]]
            lines.append(f"| {kind} | {t} | {r['median_tv']:.4f} | {cov[0]:.3f} / {cov[1]:.3f} | {min(v['holm_pvalue'] for v in q.values()):.4g} |")
    lines+=['','TV is total variation on the same finite grid; lower is better. Coverage here uses central randomized posterior-CDF intervals, with exact randomization of discrete ties. It is not the coverage of ordinary unrandomized discrete quantile endpoints. Per-quantity 68/90/95% coverage and Wilson intervals, unadjusted KS tests, and all observations/CDFs are recorded in the artifacts. Holm correction covers nine tests per method. The methods are descriptive comparisons, not a familywise claim that one is best.','',
            '![Calibration ECDF diagnostics](../results/forecast_calibration/calibration_ecdf.png)','',
            'The gray envelope is a 95% simultaneous DKW band for one ECDF; the formal decision uses the recorded Holm-adjusted tests. The data-ignoring control demonstrates why the solver-likelihood quantity is included.','',
            '| Time | Candidate corrected-minus-raw median TV, paired 95% interval | Candidate false rejection of surviving prior |','|---|---|---:|']
    for t in times:
        r=d['times'][t];ci=r['candidate_corrected_minus_raw_median_tv_ci95']
        lines.append(f"| {t} | [{ci[0]:.4f}, {ci[1]:.4f}] | {r['support']['candidate']['prior_false_rejection_mass']:.4f} |")
    lines+=['','Learned support remains part of deployment: a predicted time ratio above one rejects that parameter. This can reject true support and can admit solver-invalid parameters; both effects are recorded, without hiding them by using the solver mask for the candidate.','',
            '## Continuous-prior and numerical checks','',
            '| Time | Solver 90% coverage sigma0 / w | Candidate corrected coverage sigma0 / w | Grid mean shift, p95 in fine-grid SD | Median relative width change | Adequate |','|---|---|---|---|---|---|']
    for t in times:
        r=d['continuous'][t];f=lambda v:' / '.join(f'{x:.3f}' for x in v)
        lines.append(f"| {t} | {f(r['coverage90']['solver'])} | {f(r['coverage90']['candidate_corrected'])} | {f(r['grid_mean_shift_sd_p95'])} | {f(r['grid_width_relative_change_median'])} | {r['grid_adequate']} |")
    n=d['numerics']
    lines += ['',f"Refined solver profile difference: maximum **{n['max_abs_profile_difference_dex']:.6f} dex**, RMSE **{n['rmse_dex']:.6f} dex**; support disagreements **{n['support_disagreement']}**. Prespecified numerical screen passed: **{n['passed']}**.",'',
              'These continuous-truth intervals use ordinary posterior quantile endpoints. They are a separate quadrature/transfer diagnostic, not the exact discrete-prior SBC null. Coarse/fine posterior comparisons check identical bounds, unlike the older two grids that differed in extent. An inadequate grid or failed numerical screen blocks continuous-prior calibration claims even if the finite-prior SBC passes. The numerical spot check varies timestep tolerance and snapshot spacing, not spatial resolution or the physical SIDM closure.','',
              '## Next decision','']
    if not d['continuous_reference_passed']:
        lines.append('Prioritize the solver-posterior integration reference. The 33-to-65 grid mean shifts reach 0.30–0.53 posterior SD at the 95th percentile, exceeding the 0.1-SD threshold; the timestep/snapshot refinement itself passes. This does not prove that the 65 grid is wrong, but it prevents claiming convergence. Compare 65 with a finer same-bounds or validated adaptive quadrature reference, resolve narrow posterior ridges and the survival boundary, and use fresh continuous-prior confirmation observations. Freeze the raw forecaster throughout that integration study. Keep these discrete-prior results as a bounded diagnostic.')
    if not any(d['candidate_can_advance'].values()):
        lines.append('The candidate does not clear every predeclared advancement gate. Retain production and diagnose the specific rejected gates from `validation.json`; lower forward error alone does not justify an inference-calibration claim.')
    else:
        lines.append('At least one candidate arm clears the predeclared screens. Next test additional initial conditions, noise levels and observation designs with a fresh protocol; verify MCMC convergence if a sampler replaces grid integration. These screens alone do not authorize a production-calibration claim.')
    lines += ['', '## Reproduction and artifacts','', '```bash',
              '.venv/bin/python scripts/forecast_calibrate.py generate --workers 16',
              '.venv/bin/python scripts/forecast_calibrate.py predict --groups calibration --device cuda:0',
              '.venv/bin/python scripts/forecast_calibrate.py calibrate',
              '.venv/bin/python scripts/forecast_calibrate.py predict --device cuda:0',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_calibrate.py evaluate',
              '.venv/bin/python scripts/forecast_calibration_report.py',
              '.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts',
              '```','',
              'Verification: 121 tests passed with one stale manuscript-count test deselected. The affected forecast/calibration tests were rerun after fixing posterior-CDF roundoff at one: excursions of only a few ulps are clipped, while material normalization errors still raise. Boundary PIT values remain part of calibration diagnostics. The synthetic likelihood smoke check, compilation and whitespace checks also pass.','',
              'Calibration refuses to overwrite its frozen record. Solver/prediction caches are verified before reuse. Generated profiles live in ignored `data/d1/forecast_calibration/`. Reviewable configuration, selections, split audit, hashes, per-observation diagnostics and figures are under `results/forecast_calibration/`; logs are `logs/f3_*`. All original checkpoints and manuscript files remain unchanged.','']
    (ROOT/'docs/FORECAST_CALIBRATION_RESULTS_2026-09-09.md').write_text('\n'.join(lines))
    print('wrote F3 report and ECDF figures')


if __name__=='__main__':main()
