# F3 — inference fixes and population calibration

2026-09-09. User-directed research, fixes and next validation stage. Manuscript deferred.

## Outcome

Retain the raw candidate and reject the selected global discrepancy correction. The raw candidate passes all nine finite-grid uniformity checks and the prespecified coverage/median-TV screens. The correction improves agreement at time 100 but worsens it at 900, where the data-dependent likelihood test rejects calibration (Holm p=0.0151). Parameter-only tests would have missed this failure.

Overall advancement gates: `{'candidate_raw': False, 'candidate_corrected': False}`. Positive/negative controls pass: **True**. Numerical solver spot checks pass, but all three same-bounds grid-resolution checks fail. Continuous-prior calibration is therefore not established. Production remains unchanged; passing a rank test is not proof of exact calibration.

## What changed and why

The standalone forecast loader now validates finite, positive normalization scales and catches nonfinite/overflowed predicted terminal times before returning horizon flags. It exposes the predicted terminal time, preserves exact initial fields, and checks finite outputs. Model weights and the learned field equation are unchanged.

Added `gravop.calibration`: stable posterior normalization with explicit rejection of empty support; a Cholesky Gaussian likelihood with mean bias, correlated covariance and its normalization term; independently selected shrinkage corrections; exact randomized posterior CDFs with tied values; coverage intervals and Holm multiple-testing adjustment. Data writes are atomic, resumed prediction caches verify their data/checkpoint hashes, and calibration/confirmation identities are audited.

The method follows [Talts et al.](https://arxiv.org/abs/1804.06788) and the [Stan SBC guide](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html). [Modrak et al.](https://arxiv.org/abs/2211.02383) motivate the data-dependent likelihood test: parameter ranks alone can miss a posterior that ignores data. [Sailynoja et al.](https://arxiv.org/abs/2103.10522) motivate uniformity diagnostics; [Hermans et al.](https://arxiv.org/abs/2110.06581) motivate checking coverage and informativeness together. [Covariance shrinkage](https://scikit-learn.org/stable/modules/covariance.html) motivates regularization; our candidate correction menu and acceptance tolerances are project choices.

## Frozen experiment

Protocol: [FORECAST_CALIBRATION_PROTOCOL.md](FORECAST_CALIBRATION_PROTOCOL.md). NFW initial conditions, Yukawa cross sections, the existing uniform log-parameter prior, 20 observed density radii and 0.0434 dex Gaussian noise. Times 100, 400 and 900 are fixed physical times, not truth-dependent fractions of terminal time. The generating prior is explicitly conditioned on the solver surviving to the observation time.

A 65x65 grid with trapezoidal prior weights supplies an exact finite-prior reference. Each time has 512 independent prior/noise simulations, 1,536 total. Both methods see the same observations. Their raw and corrected likelihoods use the same observation noise. The production comparison uses its existing forward ensemble in this common likelihood framework, not its older discrepancy-inflated MCMC likelihood.

Corrections use 256 independent continuous-prior trajectories: first 192 for fitting, remaining 64 for selection, retaining survivors at each time. A further 128 untouched continuous truths check coverage and the nested 33/65 grids on the same prior bounds. Twelve prespecified direct-solver refinements check numerical error. No test result selects a correction or retunes a neural model.

| Time | Surviving grid-prior mass | Calibration fit/selection | Continuous confirmations |
|---|---:|---:|---:|
| 100 | 0.886 | 172/58 | 114 |
| 400 | 0.663 | 117/47 | 83 |
| 900 | 0.540 | 94/37 | 69 |

Generation numerical failures: `{'grid': 0, 'calibration': 0, 'confirmation': 0, 'refinement': 0}`. Calibration and confirmation share no parameter identities with each other or the old Yukawa D1 data at float32 input precision.

| Model | Time | Selected correction | Selected minus raw held-out log score |
|---|---:|---|---:|
| candidate | 100 | bias | 0.0478 |
| candidate | 400 | none | 0.0000 |
| candidate | 900 | bias_cov_0.1 | 0.3055 |
| production | 100 | bias_cov_0.1 | 0.1180 |
| production | 400 | bias_cov_0.1 | 0.3212 |
| production | 900 | bias_cov_0.5 | 5.5506 |

The corrected mean adds the fitted solver-minus-emulator bias. Covariance is fitted to clean residuals, then observational variance is added. Shrinkage values 0.1/0.5/1 blend the full covariance toward its diagonal. Selection uses shared noisy validation residuals across options. The correction is frozen without refitting; uncertainty in the fitted correction is not itself marginalized.

## Exact finite-grid SBC

| Method | Time | Median TV to solver | 90% PIT coverage, sigma0 / w | Minimum Holm p (three quantities at this time) |
|---|---:|---:|---|---:|
| solver | 100 | 0.0000 | 0.889 / 0.896 | 1 |
| solver | 400 | 0.0000 | 0.879 / 0.881 | 0.2541 |
| solver | 900 | 0.0000 | 0.908 / 0.900 | 1 |
| candidate_raw | 100 | 0.0985 | 0.898 / 0.883 | 1 |
| candidate_raw | 400 | 0.0484 | 0.875 / 0.877 | 0.3307 |
| candidate_raw | 900 | 0.0426 | 0.902 / 0.893 | 0.7987 |
| candidate_corrected | 100 | 0.0834 | 0.891 / 0.885 | 1 |
| candidate_corrected | 400 | 0.0484 | 0.875 / 0.877 | 0.294 |
| candidate_corrected | 900 | 0.0771 | 0.906 / 0.895 | 0.01506 |
| production_raw | 100 | 0.1636 | 0.873 / 0.883 | 0.12 |
| production_raw | 400 | 0.1366 | 0.871 / 0.855 | 0.01837 |
| production_raw | 900 | 0.1117 | 0.896 / 0.881 | 0.0005545 |
| production_corrected | 100 | 0.1793 | 0.883 / 0.887 | 2.661e-06 |
| production_corrected | 400 | 0.1335 | 0.877 / 0.865 | 9.751e-05 |
| production_corrected | 900 | 0.2718 | 0.898 / 0.900 | 1.052e-30 |
| prior_only | 100 | 0.8652 | 0.877 / 0.898 | 1.867e-304 |
| prior_only | 400 | 0.8064 | 0.895 / 0.914 | 6.531e-247 |
| prior_only | 900 | 0.8376 | 0.902 / 0.902 | 9.462e-273 |

TV is total variation on the same finite grid; lower is better. Coverage here uses central randomized posterior-CDF intervals, with exact randomization of discrete ties. It is not the coverage of ordinary unrandomized discrete quantile endpoints. Per-quantity 68/90/95% coverage and Wilson intervals, unadjusted KS tests, and all observations/CDFs are recorded in the artifacts. Holm correction covers nine tests per method. The methods are descriptive comparisons, not a familywise claim that one is best.

![Calibration ECDF diagnostics](../results/forecast_calibration/calibration_ecdf.png)

The gray envelope is a 95% simultaneous DKW band for one ECDF; the formal decision uses the recorded Holm-adjusted tests. The data-ignoring control demonstrates why the solver-likelihood quantity is included.

| Time | Candidate corrected-minus-raw median TV, paired 95% interval | Candidate false rejection of surviving prior |
|---|---|---:|
| 100 | [-0.0200, -0.0108] | 0.0000 |
| 400 | [0.0000, 0.0000] | 0.0000 |
| 900 | [0.0315, 0.0372] | 0.0000 |

Learned support remains part of deployment: a predicted time ratio above one rejects that parameter. This can reject true support and can admit solver-invalid parameters; both effects are recorded, without hiding them by using the solver mask for the candidate.

## Continuous-prior and numerical checks

| Time | Solver 90% coverage sigma0 / w | Candidate corrected coverage sigma0 / w | Grid mean shift, p95 in fine-grid SD | Median relative width change | Adequate |
|---|---|---|---|---|---|
| 100 | 0.877 / 0.842 | 0.860 / 0.833 | 0.488 / 0.508 | 0.024 / 0.020 | False |
| 400 | 0.855 / 0.880 | 0.855 / 0.855 | 0.297 / 0.313 | 0.020 / 0.019 | False |
| 900 | 0.870 / 0.870 | 0.870 / 0.884 | 0.322 / 0.533 | 0.029 / 0.019 | False |

Refined solver profile difference: maximum **0.001379 dex**, RMSE **0.000257 dex**; support disagreements **0**. Prespecified numerical screen passed: **True**.

These continuous-truth intervals use ordinary posterior quantile endpoints. They are a separate quadrature/transfer diagnostic, not the exact discrete-prior SBC null. Coarse/fine posterior comparisons check identical bounds, unlike the older two grids that differed in extent. An inadequate grid or failed numerical screen blocks continuous-prior calibration claims even if the finite-prior SBC passes. The numerical spot check varies timestep tolerance and snapshot spacing, not spatial resolution or the physical SIDM closure.

## Next decision

Prioritize the solver-posterior integration reference. The 33-to-65 grid mean shifts reach 0.30–0.53 posterior SD at the 95th percentile, exceeding the 0.1-SD threshold; the timestep/snapshot refinement itself passes. This does not prove that the 65 grid is wrong, but it prevents claiming convergence. Compare 65 with a finer same-bounds or validated adaptive quadrature reference, resolve narrow posterior ridges and the survival boundary, and use fresh continuous-prior confirmation observations. Freeze the raw forecaster throughout that integration study. Keep these discrete-prior results as a bounded diagnostic.
The candidate does not clear every predeclared advancement gate. Retain production and diagnose the specific rejected gates from `validation.json`; lower forward error alone does not justify an inference-calibration claim.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/forecast_calibrate.py generate --workers 16
.venv/bin/python scripts/forecast_calibrate.py predict --groups calibration --device cuda:0
.venv/bin/python scripts/forecast_calibrate.py calibrate
.venv/bin/python scripts/forecast_calibrate.py predict --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_calibrate.py evaluate
.venv/bin/python scripts/forecast_calibration_report.py
.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts
```

Verification: 121 tests passed with one stale manuscript-count test deselected. The affected forecast/calibration tests were rerun after fixing posterior-CDF roundoff at one: excursions of only a few ulps are clipped, while material normalization errors still raise. Boundary PIT values remain part of calibration diagnostics. The synthetic likelihood smoke check, compilation and whitespace checks also pass.

Calibration refuses to overwrite its frozen record. Solver/prediction caches are verified before reuse. Generated profiles live in ignored `data/d1/forecast_calibration/`. Reviewable configuration, selections, split audit, hashes, per-observation diagnostics and figures are under `results/forecast_calibration/`; logs are `logs/f3_*`. All original checkpoints and manuscript files remain unchanged.

Follow-up (2026-09-10): [F4 adaptive integration results](FORECAST_INTEGRATION_RESULTS_2026-09-10.md) add 7,990 direct-solver nodes and fresh continuous observations. The integration cap is reached without convergence; continuous-prior calibration remains unestablished.
