# Conductivity-closure test and clock-error uncertainty: results (12 November 2026)

Protocol: `docs/CLOSURE_TEST_PREREGISTRATION.md`. Every number below is read from a record under `experiments/research_extensions/` in the archive record. This report supersedes the closure verdict and the reference uncertainties of `docs/INDEPENDENT_HALOES_AND_BARYONS_RESULTS_2026-11-07.md`; its baryon results stand.

## Particle references (`multifidelity/ladder_analysis.json`, `closure_design/score_D.json`)

| Halo | 5e4 | 1e5 | 2e5 | run scatter | half time step |
|---|---|---|---|---|---|
| B (t0) | 527.9 (2) | 503.2 (2) | 544.9 (2) | 5.9% | -1.4% (paired repeat of seed 1, B_dt2_s1; closure_design/score_E.json) |
| C (t0) | 139.1 (2) | 135.6 (2) | 139.5 (2) | 2.6% | +0.7% (paired repeat of seed 1) |
| D1 (t_dyn) | - | 69.8 (3) | 68.9 (2) | 0.5-3.9% | +0.7% |
| D2 (t_dyn) | - | 85.9 (3) | 87.9 (2) | 0.5-3.5% | +1.0% |

Counts are independent seeds. The half-time-step runs repeat a 1e5 seed and are paired tests, not replicates (run C3 was pooled as a third C seed in an earlier version of this analysis; corrected). No halo shows a resolved trend with particle number; with two seeds per level this excludes only trends of more than a few per cent, and under an N^-1/2 model halo B's reference could lie between 493 and 598. AICc (counting the fitted variance) gives the no-trend model weights 0.986 (B) and 0.987 (C). Halo B's half-time-step repeat of seed 1 (B_dt2_s1) collapses 1.4% earlier than B1 (t_x10 522.8 against 530.1 t0), inside the 5.9% run scatter. References: B 525.1 (statistical 2.4%), C 138.0 (1.1%), D1 68.9, D2 87.9, each with 3% added for marker and resolution systematics.

Energy (`closure_design/energy_drift.json`, written by `scripts/energy_drift.py` from NSphere's energy histories; all 32 runs of haloes B, C, D and E): total energy is conserved to 0.06-0.31% up to t_x10; afterwards, in runs that go deep into collapse, the drift reaches up to 51%. Only the marker is used.

## Closures against the references (`learned_closures/closure_fit5.json`)

Collapse-time error, t_fluid / t_ref - 1 (%). Halo A is scored by the time rescaling that best registers its profiles; B-D by t_x10.

| Closure | A | B | C | D1 | D2 | chi2 |
|---|---|---|---|---|---|---|
| harmonic (1, 0.712) | -0.9 | +16.5 | +3.0 | -19.6 | -8.2 | 57.4 |
| van den Bosch and Tokayer (1/2, 0.9) | -2.3 | +0.3 | +7.5 | +10.1 | +9.1 | 18.2 |
| fitted (0.58, 0.85) | -4.7 | +2.2 | +3.0 | -0.9 | +1.4 | 3.95 (3 dof) |
| fitted, halo left out | -6.9 | +4.9 | +4.2 | -1.9 | +2.1 | |

alpha = 0.58 [0.555, 0.61], C = 0.85 [0.82, 0.88]: one-parameter profile intervals at Delta chi2 = 1, conditional on the adopted (diagonal) reference uncertainties. The D1 harmonic residual is 5.9 standard deviations under that model.

Direct solver check (`learned_closures/fit_direct_check.json`): at the best fit and at both ends of the alpha interval, direct runs differ from the grid interpolation by at most 1%; direct chi2 at the best fit 4.18.

Robustness (`learned_closures/closure_robustness.json`, computed in memory): alpha 0.565-0.58 and C 0.84-0.89 when the halo A uncertainty is doubled, halo A is removed, a 3% systematic is added, the systematic floor is reduced to 1%, or the smoothed particle marker is used (particle side only). Fixed-exponent baselines with C re-optimised: harmonic chi2 56.5 (C 0.70); alpha = 1/2 chi2 11.8 (C 0.94), 5.6 for the NSphere haloes alone. With the final references, A, B and C alone give alpha 0.51-0.655 (chi2_min 3.7, 1 dof). At the time D1 and D2 were designed, with the earlier B reference (545, 9.3%), they admitted alpha 0.54-1.5.

Structure before collapse (`closure_design/structure_D*.json`, seeds 21 and 22 at 1e5): density-profile RMS 0.092 / 0.060 dex (D1 / D2) for the harmonic closure, 0.039 / 0.033 dex for (1/2, 0.9).

Families (`learned_closures/loho.json`; C solved from halo B, halo C at its 2e5 mean): leave-one-out RMS 3.0% for the alpha form, 5.6% for a rational Pade form, 10.6% for a regime gate.

Closure sensitivity in Kn_c (`physics_discovery/universality_analysis.json`): a quartic in ln Kn_c fitted on 60 haloes predicts the 48 of 66 further haloes (velocity-dependent, untruncated) that lie within its fitted Kn_c range to 0.52% RMS against a 19.9% signal. Outside that range it is not valid. Fluid model only.

## Frozen-prediction test on two new haloes (`closure_design/PREREGISTRATION_E.json`, `closure_design/score_E.json`)

Frozen before any E run (record created 18:53:44 UTC and committed 18:54:27 UTC on 2026-11-12; first E run started 18:54:58 UTC): the five-halo closure (0.58, 0.85), predictions by direct solver runs, observable t_x10 in t_dyn, three 1e5 seeds, z = ln(T_obs/T_pred)/sigma_y, pass if |z| <= 2. E1: c = 19 (halo B's profile shape at about six times its cross-section), sigma-hat 1.93, Kn_c 1.12 (near D2's 1.09); E2: c = 10, a concentration of no other particle reference, sigma-hat 4.53, Kn_c 0.45 (below the fitted range). Neither entered any fit. Energy conserved to < 0.1% up to t_x10 (`closure_design/energy_drift.json`, `max_drift_to_tx10`).

| Closure | E1 error | E1 z | E2 error | E2 z |
|---|---|---|---|---|
| five-halo fit (frozen) | -6.1% | 1.5 (pass) | -11.0% | 3.4 (fail) |
| harmonic | -12.6% | 3.2 | -29.3% | 10.0 |
| (1/2, 0.9) | +1.2% | -0.3 | 0.0% | 0.0 |

At matched Kn_c (E1 vs D2) the closure errors differ by about 7%: the particle collapse depends on more than Kn_c at that level.

Seven-halo fit (post hoc, `learned_closures/closure_fit7.json`): alpha 0.535 [0.515, 0.555], C 0.886, chi2 10.0 / 5; residuals within 5.5%, leave-one-out within 9.1%. Without D: alpha 0.505; without E: alpha 0.58. Harmonic chi2 168 (144 with C re-optimised); (1/2, 0.9) chi2 18.3, 12.6 with C re-optimised. Released as `gravop.closure.CALIBRATED`; the five-halo fit as `FIVE_HALO`.

Matched markers (`learned_closures/marker_refit.json`): with t_x3, t_x10, t_x30 and an absolute threshold computed by the same code for fluid and particles, the five-halo fit gives alpha 0.555-0.575 and C 0.845-0.865; harmonic chi2 66-80 with C re-optimised.

## Clock-error propagation (`clock_uq/clock_uq.json`)

Baryon-aware emulator, three members on their own clock; sigma_c = 0.089 from 111 calibration haloes.

| Score | AUROC test (113) | AUROC held-out (81) | abstention at 90% recall | served bad |
|---|---|---|---|---|
| ensemble spread | 0.80 | 0.86 | 62% | 3.5% |
| members on own clocks | 0.87 | 0.88 | 48% | 2.7% |
| propagated | 0.95 | 0.92 | 23% | 2.1% |

Abstention is among supported queries; counting the unsupported queries (predicted phase > 1, 9.9% of test and 53% of held-out queries), which always defer, it is 31% (test) and 70% (held-out) for the propagated score. AUROC gain [0.14, 0.17] (trajectory-clustered bootstrap). The ensemble clock's median relative error on test haloes is 2.5%.

Coverage events (`clock_uq/clock_uq_events.json`, density channel, nominal 0.90, 113 test trajectories):

| Score | calibration | content coverage | half-width | simultaneous coverage | half-width |
|---|---|---|---|---|---|
| spread | same | 0.96 | 0.24 dex | 0.92 | 9.2 dex |
| propagated | same | 0.94 | 0.029 dex | 0.89 | 0.45 dex |
| propagated | split | 0.94 | 0.031 dex | 0.83 | 0.36 dex |

Content: at least 90% of a trajectory's supported queries inside the band. Simultaneous: all of them. Same: sigma_c and the conformal quantile from the same 111 calibration haloes; split: sigma_c from 55, quantile from 56. On the held-out deepest potentials content coverage is 0.37-0.47; no guarantee is claimed out of distribution. The query workload places queries at the normalised times of each trajectory (t = tau t_end); the true terminal time fixes the workload but does not enter the prediction. End-to-end gate (`clock_uq/gate_e2e.json`): 40 physical times per halo (10-1200 t0), sigma_c and the conformal multiplier from separate calibration halves. In distribution the propagated band serves 51% of queries at 0.05 dex and 65% at 0.1 dex, with 0.4% of served answers violating the budget; the spread band serves 0.8% and 33%. On the deepest potentials 15-22% of served answers violate the budget. Cost: about the solver's when queries span each halo's lifetime (one solver run per halo is almost always needed); 2.0x and 2.8x cheaper for one query per halo.
