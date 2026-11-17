# Track B results: disjoint-system transfer of the quadrature rule (2026-11-06)

Protocol: `docs/DISJOINT_PREREGISTRATION.md` (frozen before any production trajectory). Campaign
`results/track_b_20261106` (unit `sidm-track-b-20261106.service`, finished 14:41 UTC, every job
attempt 1, no failure). Report: `tasks/final_report/REPORT_FINAL.md`; numbers: `tasks/score/track_b.json`.

## Validation before scoring

* The per-trajectory decomposition reproduces `results/statecond/decomposition.json` for the five
  original FNO points to <= 5e-8 relative, and `sc_decomposition.run` to <= 9e-8; closure 1 +- 3e-9.
* Generation: 600 draws per system; kept quench/pme/cgl/hj 600, corr 588, bimodal 451.
* Every system trained under the frozen recipe (oracle nRMSE 0.005-0.157, highest in corr); converged clocks reach
  0.011-0.038% median error on Tier 1 and 0.24-0.25% on Tier 2 (published 200-epoch recipe: 0.5-6.3%). Decomposition closes everywhere (identity vs measured r: 1.9e-8).

## Results (FNO, converged clock, seeds aggregated)

| System | tier | p | p_hat | r (rule) | r (constant) | r (no penalty) | c | k | g |
|---|---|---|---|---|---|---|---|---|---|
| quench | 1 | 1.034 | 1.146 | +0.045 | +0.161 | -0.015 | -0.264 | 1.019 | 0.560 |
| pme | 1 | 1.318 | 1.426 | +0.034 | +0.055 | -0.120 | -0.075 | 0.913 | 1.017 |
| cgl | 1 | 1.021 | 1.018 | -0.002 | +0.166 | -0.009 | +0.092 | 0.955 | 0.189 |
| hj | 1 | 1.017 | 1.023 | +0.002 | +0.168 | -0.007 | -0.015 | 1.002 | 0.213 |
| corr | 2 | 1.082 | 1.274 | +0.071 | +0.141 | -0.034 | -0.273 | 0.930 | 0.790 |
| bimodal | 2 | 3.726 | 4.612 | +0.093 | -0.396 | -0.571 | -0.613 | 1.457 | 4.502 |

| Criterion | Result | Passed |
|---|---|---|
| Primary: Tier-1 mean abs r <= 0.10 and better than the constant | 0.021 [0.010, 0.030]; constant 0.138; advantage 0.117 [0.097, 0.141] | yes |
| Mechanistic identity | max deviation 1.9e-8 (closure tolerance 1e-6) | yes |
| Concentration (ranges of c, k over 9 points within 1.25x the original five) | c 1.78 vs 1.74 (1.02x); k unchanged | yes |
| No false trust (measured r inside the interval g gives under the original (c, k) box) | 6/6 inside | yes |
| Permutation null (F of one system with E_A of another must not beat the constant) | 0.210 vs 0.138 | yes |
| Adversarial: corr r < -0.2 with c > 0; bimodal RMS/median gap; both outside the envelope | corr r = +0.071, c = -0.27; bimodal RMS 0.085, median 0.052; both inside | **no** |

## Interpretation

* **The rule transfers.** On four equations disjoint from the five (non-periodic boundaries, a
  free boundary, a two-field complex equation, wall gradient blow-up), with nothing tuned, the
  coefficient-free rule predicts the converged penalty to 0.021 dex on average, and (c, k) stay in
  the envelope of the original five, so g alone gave a correct interval in every case.
* **Power caveat.** The converged clocks are so accurate that three Tier-1 systems have almost no
  penalty (p = 1.02-1.03); there the free "no penalty" rule also does well. Post hoc
  (`results/track_b_20261106/posthoc_no_penalty_comparison.json`, not preregistered): the rule still
  beats no-penalty on Tier 1 (0.021 vs 0.038; difference interval [0.004, 0.032]) and clearly on the
  three systems with a real penalty (pme, corr, bimodal: 0.066 vs 0.242 no-penalty, 0.198 constant).
* **Both designed failures failed to fail.** The adversarial predictions were wrong: the
  co-directed drift gave a negative cross term (c = -0.27), like rd, not a positive one, and the
  heavy-tailed film variant (p = 3.7, the largest penalty in the study) is predicted to 0.093 dex
  with no RMS/median gap. The rule is more robust than the plan's mechanism predicted; the plan's
  account of when it should break is refuted for these two constructions, and a construction that
  makes c large and positive remains to be found.
* **Use in the paper.** The study is reported in the manuscript's section on transfer to equations
  outside the benchmark, with the power caveat and with the no-penalty comparison labelled as post hoc.
