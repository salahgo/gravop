# F5 — solver accuracy and higher-order integration (2026-09-10)

Tighter solver settings pass the targeted temporal screen. **Higher-order integration pilot: failed.** This study does not establish population calibration. All model checkpoints, production and manuscript remain unchanged.

## Research and decisions

Protocols: [temporal diagnostics](FORECAST_NUMERICS_PROTOCOL.md) and [subsequent numerical pilot](FORECAST_NUMERICS_PILOT.md). The pilot was frozen after temporal results and before its new solver evaluations. F4’s failed result and original limits are preserved.

A positive nested Simpson rule reuses nodes while improving polynomial accuracy. [SciPy documents cubic exactness for equally spaced Simpson rules](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html). Our tests independently verify tensor-cubic exactness, fourth-order convergence, a thin diagonal ridge against an analytic integral, integer-node seeding/full prior area, invalid-mass rejection and the need to check likelihood CDFs even when moments stabilize. Raw embedded differences are retained; no division by 15 is used to relax the error screen.

[SciPy’s cubature documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html) describes higher-order adaptive subdivision without guaranteeing convergence. Its [integration tutorial](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) warns about narrow missed regions. The pilot therefore retains a global survey and seeds known peak locations; it still requires explicit posterior convergence.

## Targeted solver findings

Selected eight difficult F4 cells and 66 unique nodes before refinement. Ran 132 tighter trajectories, at eps_t=1e-4 and 3e-5 with progressively denser snapshots. Neither level had numerical failures or changed queried-time survival classifications.

| Comparison | Maximum profile change (dex) | RMS change (dex) | Temporal screen |
|---|---:|---:|---|
| 3e-4 → 1e-4 | 0.010646 | 0.001696 | False |
| 1e-4 → 3e-5 | 0.003778 | 0.000606 | True |
| 3e-4 → 3e-5 | 0.014415 | 0.002302 | False |

The local 0.00434-dex screen is one tenth of the observational noise. These outcomes support eps_t=1e-4 for a pilot, rather than reusing the coarser F4 reference. They do not prove spatial convergence or uniform temporal accuracy over the prior.

Sharp parameter variation remains after refinement. In minimum-size cell 2748 at time 400, the maximum log-density span changes from 0.178740 to 0.185114 dex. Reducing timestep error leaves this narrow feature unresolved by the quadrature. Full per-cell spans are in `audit.json`.

![Temporal and integration diagnostics](../results/forecast_numerics/numerics.png)

Frozen-model errors on these deliberately selected cells, using only rows supported by both models and both solver levels:

| Time | Common rows | Candidate RMSE to finest (dex) | Production RMSE to finest (dex) | Candidate / production rejected finest-solver survivors |
|---|---:|---:|---:|---|
| 100 | 66 | 0.002294 | 0.008092 | 0 / 0 |
| 400 | 45 | 0.014402 | 0.044461 | 4 / 9 |
| 900 | 8 | 0.037774 | 0.073453 | 0 / 0 |

These targeted results are not population error estimates. The support rejections identify a clock-boundary issue worth tracking separately from density error. The comparison uses a shared row mask; rejected survivors are reported rather than silently included in the accuracy comparison.

## Independent analytic comparison

Both methods begin with the same 4,225 nodes, use the same 0.005 raw estimated-error target and a 20,000-node cap, and retain the same unit-square prior. The examples have known exact integrals. This separate benchmark does not establish SIDM performance.

| Integrand | Rule | Nodes | Estimated relative error | Actual relative error | Target reached |
|---|---|---:|---:|---:|---|
| diagonal_ridge | trapezoid | 19985 | 0.029775 | 0.00038388 | False |
| diagonal_ridge | simpson | 19713 | 0.0047628 | 9.8574e-06 | True |
| smooth_peak | trapezoid | 7073 | 0.0049661 | 0.00049185 | True |
| smooth_peak | simpson | 5029 | 0.00074501 | 2.0464e-05 | True |
| survival_triangle | trapezoid | 5901 | 0.0042704 | 0.001103 | True |
| survival_triangle | simpson | 8593 | 0.004777 | 0.0026744 | True |

Simpson improves the smooth examples, but requires more nodes and has larger actual error on the discontinuous survival triangle. Higher order alone is therefore not a general solution to survival boundaries. The conservative error estimate can also substantially exceed measured error; it remains a diagnostic estimate, not a rigorous bound.

## Bounded integration pilot

The six cases are the worst and median F4 integration-error observations at each of times 100/400/900. These are selected, reused diagnostic observations. Their likelihood reference at the true parameters is recomputed using the tighter solver; they are not fresh tighter-solver simulations and no SBC uniformity tests are performed.

All 4060 bank nodes use eps_t=1e-4; 66 were reused from the temporal study. F4 profiles only locate initial seed points. The complete prior and survival support are retained. Stop: **node or resolution cap**. Tolerance-level convergence: **False**. Compared rules: `['pilot_initial.npz', 'pilot_latest.npz']`.

| Case | Time / selection | Final embedded error | Solver mean shifts / SD | Solver CDF changes | Candidate CDF changes | Candidate TV to solver |
|---:|---|---:|---|---|---|---:|
| 25 | t=100, worst F4 error | 0.3426 | 0.0892 / 0.1042 | 0.0000 / 0.0000 / 0.4869 | 0.0000 / 0.0000 / 0.5773 | 0.0819 |
| 27 | t=100, median F4 error | 0.2902 | 0.0058 / 0.0031 | 0.0106 / 0.0038 / 0.0006 | 0.0136 / 0.0059 / 0.0004 | 0.1206 |
| 353 | t=400, worst F4 error | 1.191 | 0.0888 / 0.0208 | 0.0781 / 0.0781 / 0.0983 | 0.1859 / 0.1859 / 0.0618 | 0.1578 |
| 310 | t=400, median F4 error | 0.2053 | 0.0414 / 0.0685 | 0.0131 / 0.0155 / 0.0492 | 0.0090 / 0.0115 / 0.0562 | 0.0776 |
| 471 | t=900, worst F4 error | 1.753 | 0.5367 / 0.5017 | 0.2404 / 0.2404 / 0.0588 | 0.2175 / 0.2174 / 0.4099 | 0.5403 |
| 536 | t=900, median F4 error | 0.2422 | 0.0075 / 0.0071 | 0.0009 / 0.0063 / 0.0110 | 0.0026 / 0.0055 / 0.0049 | 0.0716 |

Required targets are sequential estimated errors <=0.02 then <=0.005 (predefined 0.00125 fallback); every selected case must also satisfy mean change <=0.1 final SD, relative interval-width change <=0.1, and CDF change <=0.02 for both parameters and the solver likelihood. Solver and candidate are checked separately. Initial-to-final comparisons when tolerance stages are unavailable are descriptive only.

New-mode temporal check: 6 level-2 reruns, maximum profile change 0.000509 dex; support disagreements `[]`; passed **True**.

This is not a matched-budget comparison to F4: it changes solver accuracy, quadrature order, seeding and the number of target observations. Posterior comparisons remain conditional on numerical convergence.

Do not proceed to broad fresh calibration on this numerical configuration. Preserve the tighter reference and all failed checks. The integration or newly probed temporal limits identified here must be resolved before another population claim.

## Revised recommendation

Retain the tighter solver settings for the next numerical pilot. Test conditional one-dimensional integration in log sigma0 at fixed log w, using an effective-cross-section or clock coordinate to locate narrow likelihood regions. Preserve the full original prior and all branches; explicitly split at survival transitions and posterior-CDF discontinuities. Adaptation should target the CDF integrals themselves as well as evidence and moments. The large observed CDF shifts show that a small error in evidence alone would be insufficient.

This coordinate choice is a research hypothesis grounded in the project’s physics: [Outmezguine et al.](https://arxiv.org/abs/2204.06568) describe approximate temporal universality using a scattering timescale and averaged cross section. That result does not guarantee an exact mapping for this code’s finite stopping criteria or observation likelihood. Any coordinate transformation must retain the prior measure and include its Jacobian. [QUADPACK through SciPy quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html) accepts explicit breakpoints at local difficulties, which supports testing a boundary-split conditional method. This alternative is recommended, not yet implemented or validated.

For subsequent model development, retain difficult near-termination cases and report clock support errors separately from density errors. The four candidate clock rejections found here motivate a future boundary-focused training/evaluation set. These selected diagnostic cases must not become the next independent confirmation set.

## Verification and reproduction

The independent audit verifies protocol/data/source/checkpoint hashes, the reused level-1 cache, positive weights over the entire prior, mesh/error reproduction, and matrix likelihoods against direct residuals. Runtime versions and all cell spans are recorded in `audit.json`. The full suite passes 130 tests with one deferred manuscript-count check deselected (`logs/f5_tests.log`); all six added targeted tests pass.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_numerics.py --workers 20
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_numerics_pilot.py adaptive --workers 20
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_numerics_pilot.py evaluate --workers 20 --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_quadrature.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/forecast_numerics_report.py
.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts
```

Artifacts: `results/forecast_numerics/`; ignored solver caches: `data/d1/forecast_numerics/`; logs: `logs/f5_*`. Node caps are part of the frozen pilot and cannot be treated as convergence.

Follow-up: [F6 conditional integration results](FORECAST_CONDITIONAL_RESULTS_2026-09-10.md)
now test the proposed alternative on one difficult interior-time case. Inner
columns and the stricter temporal check pass, but outer parameter-CDF error
exceeds tolerance at the fixed budget. F5's negative result remains unchanged.
