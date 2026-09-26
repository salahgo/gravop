# F6 — conditional posterior integration (2026-09-10)

**Numerical pilot passed: False.** This is one selected diagnostic observation, not a population calibration result. Production, model weights and manuscript remain unchanged.

## Method and scope

Frozen protocol: [FORECAST_CONDITIONAL_PROTOCOL.md](FORECAST_CONDITIONAL_PROTOCOL.md). F4 observation 353 at time 400 is a difficult interior-time case from F5. Its observations and tighter-solver truth likelihood are reused for numerical development, with the complete original prior and observational noise unchanged.

The conditional method surveys each sigma0 slice, locates both methods’ likelihood modes and shoulders, brackets survival transitions, and splits at parameter and solver-likelihood CDF boundaries. Positive Gauss–Legendre 3/5 rules control inner evidence, moment and CDF errors. Positive nested Simpson rules integrate over w and propagate the inner uncertainties; the w-CDF is classified per interval to avoid artificial endpoint atoms.

Explicit breakpoints follow the numerical principle supported by [QUADPACK through SciPy quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html). [Outmezguine et al.](https://arxiv.org/abs/2204.06568) motivate examining clock-related narrow regions, but this implementation locates peaks directly and makes no exact universality assumption or prior-coordinate transformation. A finite survey cannot prove the absence of arbitrarily narrow hidden modes or survival islands.

All direct reference values use the tighter level-1 solver. Candidate predictions use frozen checkpoints on CPU. No interpolated forward reference, tail truncation or discrepancy fit is introduced. Per-column caches save every four new solves and source/data/model identities are checked on resume.

## Numerical result

New level-1 trajectories: **8000**. Stop: `global trajectory budget exhausted`. Joint numerical convergence: **False**. Caps were 8,000 new trajectories overall and 512 per column.

The global 8,000-trajectory budget was exhausted while a final column had completed 106 new solves; its per-column allowance was limited by the remaining global budget. This was a budget stop, not a reported direct-solver failure. 29 of 30 attempted columns passed their inner checks. The last complete outer rule uses 25 columns at 6514 new trajectories; the subsequent 1486 evaluations did not yield another complete rule. Neither tolerance stage was reached.

The limiting quantity in both methods is the sigma0 CDF numerator. In the last complete rule, most of its error comes from log-w [0.6492923469631748, 1.5]. Additional completed slices show the solver conditional sigma0 CDF rising from 0.268931 at log-w 0.64929235 to effectively 1 at 0.75563080. Explicit inner CDF cuts do not resolve this outer transition. The anticipated inherited-node cache miss also occurred, consuming 338 additional evaluations for the one-ULP-shifted column.

![Conditional integration diagnostics](../results/forecast_conditional/conditional.png)

| Stage | Outer panels | Columns | New solves | Solver maximum error | Candidate maximum error |
|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 13 | 3839 | 0.0779297 | 0.137913 |
| 0 | 4 | 17 | 4804 | 0.0458093 | 0.104092 |
| 0 | 5 | 21 | 5567 | 0.0338561 | 0.0619675 |
| 0 | 6 | 25 | 6514 | 0.0339683 | 0.0340014 |

| Method | Mean (log sigma0 / log w) | 90% widths | CDFs (sigma0 / w / solver likelihood) | TV to solver |
|---|---|---|---|---:|
| solver | -0.14992 / 0.58606 | 0.91688 / 1.66232 | 0.45882 / 0.51733 / 0.65055 | 0.00000 |
| candidate | -0.12575 / 0.57842 | 1.08594 / 1.74045 | 0.46567 / 0.51092 / 0.66056 | 0.14544 |

On this rule, solver posterior mass outside candidate survival support is 0; candidate posterior mass outside solver support is 1.69688e-193. The quadrature estimate of prior mass with differing survival classifications is 0.000471364. These are diagnostic estimates on the reported rule and remain subject to its numerical gate.

Joint targets are 0.02 then 0.005, with a third 0.00125 level available if prescribed by the successive-summary checks. Inner tolerances are one tenth of these targets. Every outer interval is bisected between stages. Both methods require mean changes <=0.1 final SD, relative 90%-width changes <=0.1, and all three CDF changes <=0.02. Positive-node quantile widths are numerical approximations; their convergence is checked separately.

The following errors are normalized by each method’s joint evidence. Each entry is the maximum over the eight controlled quantities; maxima from different columns need not refer to the same quantity.

| Outer log-w interval | Solver outer error | Solver propagated inner error | Candidate outer error | Candidate propagated inner error |
|---|---:|---:|---:|---:|
| -1.00000 / -0.68750 | 3.33547e-85 | 3.37088e-89 | 7.73096e-82 | 1.13015e-85 |
| -0.68750 / -0.53125 | 0.00331331 | 4.93997e-06 | 0.00736253 | 1.14602e-05 |
| -0.53125 / -0.37500 | 8.7302e-07 | 4.52169e-05 | 3.62243e-05 | 4.76034e-05 |
| -0.37500 / 0.25000 | 0.00886635 | 0.000264993 | 0.00768357 | 0.000284592 |
| 0.25000 / 0.64929 | 0.00497571 | 6.4739e-05 | 0.00494952 | 5.88059e-05 |
| 0.64929 / 1.50000 | 0.0289231 | 7.413e-05 | 0.0289022 | 0.000153712 |

The direct peak search can be compared with the preceding F5 grid wherever exactly the same w column exists. Both sides use the same tighter solver and observation:

| log w | Best F5 cached log likelihood | Best F6 cached log likelihood |
|---:|---:|---:|
| -1.000000 | -203.168719 | -203.168719 |
| -0.921875 | -230.372837 | -230.372837 |
| -0.843750 | -278.147731 | -278.147731 |
| -0.765625 | -348.043730 | -348.043730 |
| -0.687500 | -439.326374 | -439.326374 |
| -0.648438 | -824.814443 | -492.163625 |
| -0.609375 | -434.184270 | -434.184270 |
| -0.570312 | -40.782152 | -40.782152 |
| -0.531250 | -14.269108 | -12.862856 |
| -0.492188 | -20.930511 | -12.815752 |
| -0.453125 | -52.709865 | -12.768366 |
| -0.414062 | -38.547839 | -12.720468 |
| -0.375000 | -34.572058 | -12.672302 |
| -0.218750 | -17.824281 | -12.481254 |
| -0.062500 | -54.035661 | -12.306221 |
| 0.093750 | -49.921086 | -12.164847 |
| 0.250000 | -18.603526 | -12.064540 |
| 1.500000 | -12.459542 | -11.911594 |

These differences show which sampled likelihood peaks were previously missed; a better sampled maximum alone does not establish a converged posterior integral.

Additional level-2 check: 5 solver peaks selected through evenly spaced outer posterior-mass quantiles, maximum profile change 0.000454454 dex, survival disagreements `[]`, passed **True**. Selection was recorded before these stricter runs.

At the same selected peaks, candidate density RMSE to the level-2 reference at time 400 is 0.00246404 dex (0.05678 times the observational noise), with maximum absolute difference 0.00696265 dex across 5 common-survival profiles. This is a targeted model diagnostic on the reused case, not an independent population accuracy estimate.

## Interpretation and verification

A separate [development note](FORECAST_CONDITIONAL_FOLLOWUP_2026-09-10.md) examines outer prior-boundary, parameter-CDF and likelihood-CDF features, documents the cache-reuse issue, and reports 24 analytic benchmark runs. Those benchmarks do not replace this direct-solver pilot or change its frozen protocol.

The configuration does not yet support broader calibration. Preserve all failed gates and diagnose the specific unresolved conditional or outer integration error before expanding the study. The existing candidate model has not been tuned on these observations.

A read-only cache audit also found an efficiency issue: bisecting the original upper outer panel regenerates an inherited node as 0.8619692602223811 instead of 0.8619692602223812. Exact-byte cache keys treat these as separate columns. The new helper `src/gravop/dyadic.py` uses fixed base bounds and integer descendant indices to preserve inherited coordinates exactly. Three tests reproduce the cache miss, verify bit-identical reuse through 17 refinement depths and shared base endpoints, and reject invalid or unrepresentable panels (`logs/f6_dyadic_tests.log`). Integration into a subsequent driver remains outstanding. This run retains its frozen arithmetic and counts all actual trajectories; the finding does not change a failed numerical gate or authorize merging distinct cached coordinates. Details: `results/forecast_conditional/cache_roundoff_diagnostic.json`.

The audit verifies frozen input/source/checkpoint hashes and, when an outer rule exists, positive weights covering the full prior, propagated error reproduction, direct likelihood arithmetic, integrated means and CDFs. Analytic tests cover off-grid narrow peaks, multiple modes with disconnected support, candidate mass outside solver support, explicit outer CDF sides, inner-error propagation and exact inherited-node reuse. The final full suite passed 138 tests; output is in `logs/f6_tests_verified.log`. One manuscript-count test remains deferred.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_conditional.py --workers 16
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_conditional_report.py
.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts
```

Artifacts: `results/forecast_conditional/`; ignored caches: `data/d1/forecast_conditional/`; logs: `logs/f6_*`. Earlier F4/F5 failures remain recorded.
