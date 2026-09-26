# F7 — posterior convergence repair (2026-09-10)

**Overall numerical and temporal validation passed: True.** One reused diagnostic case at time 400; this is not population calibration. Models, production and manuscript remain unchanged.

Protocol: [FORECAST_POSTERIOR_REFINE_PROTOCOL.md](FORECAST_POSTERIOR_REFINE_PROTOCOL.md). The failed [F6 experiment](FORECAST_CONDITIONAL_RESULTS_2026-09-10.md) is preserved.

The revised outer tree uses fixed base bounds and integer dyadic descendants, so inherited nodes retain exact cache identities. Observed parameter-CDF, likelihood-CDF and evidence changes guide initial branch refinement. Complete verified inner partitions are reused at tighter tolerances. Already-required survey and quadrature evaluations are batched through a shared solver pool; the full prior, both survival supports and direct likelihoods are retained.

The design follows the use of explicit difficult-feature breakpoints in [QUADPACK](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html), while retaining custom positive vector rules and propagated inner errors. A finite discovery survey is not a proof against arbitrary hidden modes or support islands.

Stop: `successive tolerance-level comparison`. Reserved new level-1 submissions: 28866; recorded completed evaluations: 28848. The separately frozen ceiling is 80,000 submissions, 1,024 per column. Preflight cost accounting increased the unused preliminary ceiling before any F7 solver submissions; those preliminary records are preserved.

Execution started with 24 solver workers and 12 column threads, then resumed from durable caches with 24 column threads after a concurrency check. The reservation ledger retains 18 submissions without recorded completed evaluations across that restart; they were not refunded. See `concurrency_restart.json` and `process.json`. Numerical sources and acceptance gates stayed fixed.

## Convergence checks

| Stage | Panels | Columns | New submissions | Solver maximum error | Candidate maximum error |
|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 61 | 10380 | 0.002602822 | 0.007454919 |
| 1 | 30 | 121 | 28866 | 0.001477018 | 0.0009644452 |

![Convergence](../results/forecast_posterior_refine/convergence.png)

The unchanged targets are 0.02, 0.005 and optional 0.00125; inner tolerances are one tenth as large. Every outer panel is bisected between stages. Both methods must satisfy mean shifts <=0.1 final SD, relative 90%-width changes <=0.1 and all three CDF changes <=0.02. Marginal widths remain positive-node approximations and are explicitly checked for stability.

| Method | Largest mean shift (SD) | Largest relative width change | Largest CDF change | Passed |
|---|---:|---:|---:|---|
| solver | 0.004545078 | 0.0291845 | 0.000563921 | True |
| candidate | 0.001898831 | 0.02798446 | 0.0008551881 | True |

| Method | Mean (log sigma0 / log w) | 90% widths | CDFs (sigma0 / w / solver likelihood) | TV to solver |
|---|---|---|---|---:|
| solver | -0.144914 / 0.581021 | 0.918376 / 1.82183 | 0.475605 / 0.519454 / 0.643271 | 0 |
| candidate | -0.11867 / 0.571234 | 1.09236 / 1.89996 | 0.481266 / 0.514001 / 0.654656 | 0.143563 |

Temporal check: 5 selected peaks, 4 new level-2 evaluations, maximum profile change 0.000454574 dex, support disagreements `[]`, passed **True**. Selection was saved before new evaluations.

## Audit and interpretation

The audit independently reproduces positive full-prior weights, cached solver likelihoods, vector integrals, propagated errors, means, widths and CDFs, and rechecks frozen sources/data/checkpoints. When tolerance stages finish, their recorded comparison is recomputed. Solver reservations are durable and exclusive-run locking prevents duplicate drivers.

Test result: `142 passed, 1 deselected in 170.38s (0:02:50)`. The manuscript-count check remains deferred. Analytic tests include a complete two-method outer integration of curved Gaussian ridges with exact conditional moments/CDFs and independent reference integrals, reuse of verified inner partitions, exact outer-node identities and concurrent budget reservations. Concurrency checks matched serial frozen-candidate outputs bit-for-bit for 36 predictions with 12 threads and 72 predictions with 24 threads.

The repaired method clears the prescribed numerical gates for this case. This supports extending the same validation to other difficult observations and times before fresh population calibration. It does not establish population calibration or justify production replacement.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_posterior_refine.py --workers 24 --columns 24
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_posterior_refine_report.py
```

Artifacts: `results/forecast_posterior_refine/`; caches: `data/d1/forecast_posterior_refine/`; logs: `logs/f7_*`. Earlier failed experiments remain unchanged.
