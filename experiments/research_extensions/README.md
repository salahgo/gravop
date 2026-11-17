# Conductivity-closure test and clock-error uncertainty

Code for two studies in the manuscript:

- the test of the gravothermal conductivity closure against NSphere particle simulations (Section 4), including the pre-registered haloes D1, D2 and the frozen-prediction test on E1, E2;
- the clock-error propagation for the emulator's uncertainty (Section 6.2).

The result records and the per-run particle measurements are in the archive record, at the same paths under `experiments/research_extensions/`. Unpack it at the root of a checkout and every script below runs from stored data unless it is marked otherwise; `python scripts/reproduce.py closure_test --execute` runs the stored-data stages in order. The matched-marker refit reads the fluid grid `learned_closures/closure_scan_traj.npz`, which is archived.

| Step | Script | Writes |
|---|---|---|
| Particle references (needs NSphere, see below) | `scripts/nsphere_ext.py launch <run> --threads T`, then `scripts/nsphere_ext.py measure <run>` | `<group>/nsphere/<run>/{run.json, measures.npz}` |
| Particle-number ladders | `scripts/multifidelity_analysis.py` | `multifidelity/ladder_analysis.json` |
| (alpha, C) scans of the fluid model | `scripts/closure_scan.py --workers 4`, `scripts/closure_scan_D.py` | `learned_closures/closure_scan*.json` |
| Identifiability of A, B, C | `scripts/closure_identifiability.py` | `learned_closures/identifiability_*.json` |
| Design of D1, D2 | `scripts/design_scan.py`, `scripts/design_select.py` | `closure_design/design_scan.json`, `closure_design/PREREGISTRATION_D.json` |
| Frozen forecast | `scripts/family_posterior.py`, `scripts/c_sensitivity.py`, `scripts/forecast_D.py` | `collapse_probability/FORECAST_D.json` |
| Scoring of D1, D2 | `scripts/score_D.py` | `closure_design/score_D.json` |
| Fluid convergence of the D predictions | `scripts/fluid_convergence_D.py` | `closure_design/fluid_convergence_D.json` |
| Structure along the evolution (pinned to seeds 21, 22; `--all` for every run) | `scripts/compare_D_structure.py D1` (and `D2`) | `closure_design/structure_D*.json` |
| Five-halo fit, leave-one-out | `scripts/closure_fit5.py` | `learned_closures/closure_fit5.json` |
| Robustness of the fit (in memory; does not rewrite the fit record) | `scripts/closure_robustness.py` | `learned_closures/closure_robustness.json` |
| Direct solver check of the fit (8 cores, minutes) | `scripts/fit_direct_check.py` | `learned_closures/fit_direct_check.json` |
| Particle energy at the marker (needs raw dumps) | `scripts/energy_drift.py` | `closure_design/energy_drift.json` |
| Pre-registration of family E, halo B half-step control | `scripts/design_select_E.py` | `closure_design/PREREGISTRATION_E.json` |
| Scoring of E, B half-step | `scripts/score_E.py` | `closure_design/score_E.json` |
| Seven-halo fit (post hoc) | `scripts/closure_fit7.py` | `learned_closures/closure_fit7.json`, `closure_scan_E.json` |
| Matched-marker refit | `scripts/marker_refit.py` | `learned_closures/marker_refit.json` |
| Pade and gate families | `scripts/gfamilies.py`, `scripts/loho_closures.py` | `learned_closures/loho.json` |
| Closure sensitivity in Kn_c | `scripts/universality.py 3`, `scripts/universality_analysis.py` | `physics_discovery/universality*.json` |
| Clock-error propagation (GPU; reads the archived baryon-aware members and dataset) | `scripts/clock_uq.py` | `clock_uq/clock_uq.json` |
| End-to-end gate on a physical-time workload (GPU; cost ratios are estimates from stored solver times and measured emulator throughput) | `scripts/gate_e2e.py` | `clock_uq/gate_e2e.json`, per-halo deferral and solver times in `clock_uq/gate_e2e.npz` |
| Coverage events and total fallback (GPU) | `scripts/clock_uq_events.py` | `clock_uq/clock_uq_events.json` |
| Emulator retrained on the five- and seven-halo closures (GPU) | `scripts/closure_fit_emulator_eval.py` at the repository root (target `closure_fit7_emulator`) | `results/closure_fit_20261112/emulator.json`, `results/closure_fit7_20261113/emulator.json` |
| Tabulated-closure solver test | `pytest tests/test_gclosure.py` | |

The manuscript figures of these studies are drawn by `scripts/fig_closure_test.py`, `scripts/fig_closure_calibration.py`, `scripts/fig_closure_structure.py` and `scripts/fig_clock_uq.py` at the repository root.

## NSphere

NSphere (Kamionkowski and Sigurdson, arXiv:2504.13996; Apache-2.0) is not redistributed. The runs used commit `8f14666` built in a Docker image; `closure_design/nsphere_provenance.json` records the image digest, binary hash and compiler, and every `run.json` records the build line, the exact command, the seed, the step count and the wall time. NSphere adjusts the number of steps unless (steps - 1) is a multiple of the output interval, so `nsphere_ext.py` fixes both. The raw particle dumps (about 40 GB) are not archived; `measures.npz` holds the Lagrangian radii, mean densities, dispersions and density profiles measured from them.

## Order of the pre-registered test

`closure_design/PREREGISTRATION_D.json` records its creation time (2026-11-11 23:12:51 UTC) and a SHA-256 of its content. The first D run started at 23:44:43 UTC (`closure_design/nsphere/D1_s21/run.json`); the scoring formula was fixed while that run was in progress, before any run reached its marker. See `docs/CLOSURE_TEST_PREREGISTRATION.md`.
