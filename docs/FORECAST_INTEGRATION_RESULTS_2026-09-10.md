# F4 — adaptive posterior integration (2026-09-10)

Integration gates passed: **False**. Solver/negative controls passed: **True**. Raw candidate passed all numerical/statistical gates: **False**. Production and manuscript remain unchanged.

## Method and scope

Protocol: [FORECAST_INTEGRATION_PROTOCOL.md](FORECAST_INTEGRATION_PROTOCOL.md). The new positive dyadic quadrature reuses the 65x65 full-prior survey and spends direct-solver evaluations on cells with large normalized embedded error. Each cell compares corner trapezoids against its 3x3 fine rule. The sum of absolute local differences is an error estimate, not a rigorous bound. No emulator or interpolated forward field supplies reference-node values.

This follows the global adaptive-subdivision idea described in [SciPy cubature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html), with a custom reusable-node rule. The [SciPy integration tutorial](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) warns about missing narrow important regions. We keep the full prior survey and check successive posterior means, widths and CDF values before interpreting calibration.

Generated 256 independent continuous-prior trajectories: survivors [227, 172, 144] at fixed times 100/400/900, 543 noisy observations in total. Solver failures: 0; earlier-data overlaps: `{'F3_calibration': 0, 'F3_confirmation': 0, 'D1': 0}`. The existing NFW/Yukawa prior, 20 radii and 0.0434 dex observational noise remain fixed. The population is explicitly conditioned on survival at each time. No model or discrepancy correction was fitted to these observations.

The raw candidate and production forward ensemble use the same observational-noise-only likelihood; this does not re-evaluate the old production discrepancy-inflated MCMC pipeline. The rejected F3 correction remains disabled.

## Numerical outcome

Final solver-node count: **12215**, of which **7990** are new adaptive evaluations. Stopping status: `refinement cap reached; do not claim integration convergence`. Maximum final embedded relative error: **1.30696**. Compared rules: `['initial_rule.npz', 'latest_rule.npz']`.

![Adaptive integration progress](../results/forecast_integration/integration.png)

| Method | Time | p95 mean shift / final SD | Median relative width change | p95 CDF changes (sigma0, w, likelihood) | Pass |
|---|---:|---|---|---|---|
| solver | 100 | 0.1352 / 0.2361 | 0.0000 / 0.0000 | 0.0751 / 0.0785 / 0.0709 | False |
| solver | 400 | 0.2606 / 0.4239 | 0.0000 / 0.0000 | 0.1326 / 0.1037 / 0.1198 | False |
| solver | 900 | 0.1239 / 0.2299 | 0.0000 / 0.0000 | 0.1088 / 0.0536 / 0.0793 | False |
| prior_only | 100 | 0.0008 / 0.0000 | 0.0000 / 0.0087 | 0.0030 / 0.0020 / 0.0017 | True |
| prior_only | 400 | 0.0008 / 0.0020 | 0.0000 / 0.0000 | 0.0036 / 0.0026 / 0.0019 | True |
| prior_only | 900 | 0.0009 / 0.0011 | 0.0000 / 0.0000 | 0.0043 / 0.0024 / 0.0017 | True |
| candidate | 100 | 0.0723 / 0.1780 | 0.0000 / 0.0000 | 0.0674 / 0.0664 / 0.0572 | False |
| candidate | 400 | 0.2605 / 0.4254 | 0.0000 / 0.0000 | 0.1359 / 0.1055 / 0.1014 | False |
| candidate | 900 | 0.1759 / 0.2947 | 0.0000 / 0.0000 | 0.1110 / 0.0610 / 0.0760 | False |
| production | 100 | 0.1413 / 0.2459 | 0.0000 / 0.0000 | 0.1166 / 0.0717 / 0.0747 | False |
| production | 400 | 0.2191 / 0.3219 | 0.0000 / 0.0000 | 0.1222 / 0.1102 / 0.0966 | False |
| production | 900 | 0.1686 / 0.2988 | 0.0000 / 0.0000 | 0.1108 / 0.0797 / 0.0665 | False |

Thresholds are p95 mean change <=0.1 SD, median relative interval-width change <=10%, and p95 CDF change <=0.02 for each quantity. Solver and candidate must both pass, together with estimated-error convergence. A budget-limited rule is never labeled converged merely because some summary statistics stabilize.

Neither tolerance level was reached. The initial-versus-final comparison above is descriptive; it does not substitute for the protocol’s required comparison between completed tolerance levels. Zero median interval-width changes reflect discrete quantile endpoints and do not establish convergence.

The recomputed final embedded-error distribution is:

| Time | Median | p95 | Maximum | Fraction below 0.02 |
|---|---:|---:|---:|---:|
| 100 | 0.1954 | 0.8198 | 1.0751 | 0.022 |
| 400 | 0.1860 | 0.8330 | 1.0564 | 0.000 |
| 900 | 0.2851 | 0.8922 | 1.3070 | 0.000 |

There are 84 cells at the minimum allowed size. Their summed normalized embedded error reaches 0.6710 for one observation. However, the worst observation at each time has no error contribution from these cells: both the node budget and local resolution deserve attention. These are estimated quadrature errors, not measured forward-model errors.

## Fresh continuous-prior diagnostics

| Method | Time | n | Median TV to solver | Central-CDF 90% coverage (sigma0 / w) | Minimum Holm p |
|---|---:|---:|---:|---|---:|
| solver | 100 | 227 | 0.0000 | 0.8811 / 0.9031 | 1 |
| solver | 400 | 172 | 0.0000 | 0.9012 / 0.9128 | 1 |
| solver | 900 | 144 | 0.0000 | 0.8750 / 0.9028 | 1 |
| prior_only | 100 | 227 | 0.8621 | 0.9383 / 0.9119 | 2.492e-148 |
| prior_only | 400 | 172 | 0.8232 | 0.9419 / 0.9186 | 4.947e-99 |
| prior_only | 900 | 144 | 0.8253 | 0.9097 / 0.9236 | 7.354e-82 |
| candidate | 100 | 227 | 0.0975 | 0.8722 / 0.8987 | 0.5366 |
| candidate | 400 | 172 | 0.0485 | 0.9012 / 0.9070 | 1 |
| candidate | 900 | 144 | 0.0446 | 0.8542 / 0.8889 | 1 |
| production | 100 | 227 | 0.1638 | 0.8811 / 0.8987 | 0.001624 |
| production | 400 | 172 | 0.1383 | 0.9070 / 0.9070 | 0.4056 |
| production | 900 | 144 | 0.1234 | 0.8681 / 0.9097 | 0.6905 |

CDF values use positive quadrature weights; the truth parameters are continuous and independently drawn. The reference likelihood quantity is evaluated from the direct-solver truth profile and the same noise draw. Numerical CDF error is checked separately. Holm adjustment covers nine tests per method. Ordinary quantile-endpoint coverage, 68/90/95% central-CDF coverage with Wilson intervals, p90 TV, widths and test statistics are in `validation.json`; per-observation summaries are in `diagnostics.npz`.

Estimated surviving prior mass at times 100/400/900: 0.8865 / 0.6626 / 0.5405. Candidate mean posterior mass outside solver support is at most 1.26e-12; mean solver posterior mass rejected by the candidate is at most 0.000148. These are also provisional mesh estimates.

Continuous-prior calibration remains unestablished because the integration gates fail. Treat the statistical diagnostics as provisional; they cannot distinguish all emulator errors from an insufficient numerical reference. Preserve the raw candidate and the cache, and resolve the remaining integration limits before a production claim.

## Next numerical experiment

Keep the frozen raw candidate. First inspect solver smoothness and survival transitions in the cells that have already reached the resolution limit, using stricter direct solves. Then benchmark a higher-order or per-observation integration rule on a fixed diagnostic subset, measuring actual posterior CDF stability as well as evidence error. SciPy’s cubature offers Gauss–Kronrod and Genz–Malik rules, but their suitability for these likelihoods is untested; they are candidates for a separate experiment, not validated replacements. Preserve this failed F4 result and its original thresholds. Use fresh confirmation draws after choosing the next numerical method; do not tune model weights or uncertainty on F4 observations.

## Reproduction and verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_integrate.py generate --workers 20
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_integrate.py adaptive --workers 20
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_integrate.py evaluate --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_integration_audit.py
.venv/bin/python scripts/forecast_integration_report.py
.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts
```

Generation refuses to overwrite the fresh observations. Adaptive computation resumes its atomic node/mesh cache. The preceding prediction cache must be archived if the bank is subsequently refined. The predefined third tolerance can be requested with `adaptive --refine-next` after a completed second stage if candidate convergence fails.

Verification: 124 tests passed, one stale manuscript-count check deselected. New analytic tests cover positive weights/full area after local subdivision, bilinear integration, separated narrow Gaussian peaks and rejection of invalid/zero-mass integrands. The independent cache audit reproduces final weights and error estimates, checks the matrix likelihood against explicit residuals, and verifies recorded protocol, observation, bank, prediction, source and checkpoint hashes. Runtime library/device versions are in `final_error_diagnostic.json`. The first interrupted refinement batch was restarted from the saved mesh; completed node batches are reused.

Artifacts: `results/forecast_integration/`; ignored solver caches: `data/d1/forecast_integration/`; logs: `logs/f4_*`. No manuscript or model checkpoint was modified.

Follow-up: [F5 solver and higher-order integration study](FORECAST_NUMERICS_RESULTS_2026-09-10.md) improves the targeted temporal accuracy screen, but its bounded integration pilot also fails. F4's numerical result remains unchanged.
