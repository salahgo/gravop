# Conductivity-closure test: pre-registration of haloes D1 and D2

Frozen record: `experiments/research_extensions/closure_design/PREREGISTRATION_D.json` (created 2026-11-11 23:12:51 UTC; the file carries a SHA-256 of its own content).

## Question

Haloes A (N-body), B and C (NSphere) do not determine the shape of the conductivity interpolation K = (K_s^-alpha + K_l^-alpha)^(-1/alpha). With the references available at the time (halo B: 545 t0, 9% uncertainty), closures with alpha between about 0.5 and 1.5 fitted all three. In the fluid model, closures that agree on A, B and C differ by 20-40% in collapse time when the central Knudsen number at the largest core is below one.

## Design

- Candidates: NSphere truncated NFW haloes over a grid of sigma-hat and concentration.
- For each candidate, every closure in the valley was run with the fluid model; the separation S = |ln T1 - ln T2| / sigma_y was computed for each pair, with sigma_y the uncertainty of a seed-mean observation (seed scatter / sqrt(n), particle-number and marker systematics). The observation noise enters once.
- Selection was made by inspecting the scan for large S at affordable particle cost (the choice is recorded in `scripts/design_select.py`, not computed by an automatic ranking).
- Selected: D1 (M = 1e10 Msun, r_s = 3 kpc, c = 6, sigma/m = 200.2 cm^2/g, Kn_c 0.60) and D2 (same profile, 113.4 cm^2/g, Kn_c 1.09).

## Frozen record

- Predicted collapse markers of six closures: harmonic (alpha 1, C 0.712), alpha 1/2 with C fitted on A (0.878), van den Bosch and Tokayer (1/2, 0.9), and three members of the valley (alpha 0.557, 0.669, 1.5).
- Observable: t_x10, the first time the mean density inside 0.3% of the mass exceeds ten times its pre-collapse minimum, seed mean, same pipeline as B and C.
- Rule: closure parameters may be fitted only on A, B and C and their ladders; D1 and D2 are held-out tests.
- Runs: three seeds at 1e5 particles, two at 2e5, one at half the time step, per halo.
- Probabilistic forecast: `experiments/research_extensions/collapse_probability/FORECAST_D.json`.
- Scoring rules (`experiments/research_extensions/scripts/score_D.py`): z = ln(T_obs / T_pred) / sigma_y per closure, and the log score and PIT of the forecast, against the 2e5-particle seed mean.

Chronology: predictions, observable, rule and run plan frozen at 23:12:51 UTC; the first D run (D1_s21) started at 23:44:43 UTC; the scoring formula was committed at about 00:22 UTC while that run was in progress (55% complete) and before any run had reached its collapse marker; the first run finished at about 00:49 UTC. Results: `docs/CLOSURE_TEST_RESULTS_2026-11-12.md`.
