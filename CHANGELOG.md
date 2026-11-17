# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released version is archived on Zenodo and carries its own version DOI;
the concept DOI always resolves to the most recent one.

## [1.2.0] - 2026-11-17

### Added
- Test of the gravothermal conductivity closure against NSphere particle
  simulations: particle-number ladders for haloes B and C, two haloes (D1, D2)
  selected by a design calculation and pre-registered before simulation, the
  five-halo fit of the generalised interpolation with leave-one-out
  prediction, its robustness study, and the closure-sensitivity analysis in the
  central Knudsen number (`experiments/research_extensions/`,
  `docs/CLOSURE_TEST_PREREGISTRATION.md`,
  `docs/CLOSURE_TEST_RESULTS_2026-11-12.md`).
- Clock-error propagation for the emulator's uncertainty
  (`experiments/research_extensions/scripts/clock_uq.py`).
- Figure scripts for the closure test and the clock-error propagation
  (`scripts/fig_closure_test.py`, `scripts/fig_closure_calibration.py`,
  `scripts/fig_closure_structure.py`, `scripts/fig_clock_uq.py`).
- Archive record: the closure-test result records and the per-run particle
  measurements (`measures.npz`, `run.json`) under
  `experiments/research_extensions/`, and the halo C 2e5-particle seed
  `results/independent_halo_20261107/nsphere_C5.npz`.

### Fixed
- Particle-number ladder: the half-time-step repeat of a halo C seed (C3) was
  pooled as an independent seed; it is now a paired time-step test, and the
  halo C reference moves from 137.5 to 138.0. AICc now counts the fitted
  variance. The closure fit reads its B and C references from the ladder record.
- `closure_robustness.py` no longer rewrites the canonical fit record.
- `compare_D_structure.py` uses a pinned run list.

### Added (completion of the open tests)
- Pre-registered test of the frozen five-halo closure on two new haloes
  (E1, E2; `design_select_E.py`, `score_E.py`): passed in range, failed
  below it (E2, 11% early). Seven-halo fit (`closure_fit7.py`):
  alpha 0.535, C 0.886, now `gravop.closure.CALIBRATED`; the five-halo fit is
  `FIVE_HALO`.
- Refit with identical marker processing for fluid and particles
  (`marker_refit.py`); end-to-end test of the solver-fallback rule on a
  physical-time workload (`gate_e2e.py`); emulators retrained on the five- and
  seven-halo closures (`scripts/closure_fit_emulator_eval.py`).
- Halo B half-time-step control (`B_dt2_s1`): t_x10 moves by -1.4% against B1, inside the seed scatter.

### Added (review follow-up)
- Direct solver check of the fitted closure (`fit_direct_check.py`), coverage
  events and total fallback of the clock-error bands (`clock_uq_events.py`),
  particle energy drift at the collapse marker (`energy_drift.py`,
  `energy_drift.json`; all 32 NSphere runs), and the
  NSphere build provenance (`nsphere_provenance.json`).

### Added (release completeness)
- Archive record: the five members of the emulator retrained on the
  recommended closure (`results/closure_fit7_20261113/runs/*/model.pt`) and
  the fluid grid read by the matched-marker refit
  (`experiments/research_extensions/learned_closures/closure_scan_traj.npz`).
- Reproduction targets `closure_test` (every stored-data stage of the closure
  test, its figures and tests; CPU) and `closure_fit7_emulator` (dataset
  regeneration and retraining under the recommended closure; GPU).
- Archive record: the three baryon-aware emulator members and their dataset
  (`results/baryons_20261107/runs/seed*/model.pt`, `data/baryons/train.h5`),
  so the clock-error, coverage and gate studies replay exactly; per-halo
  deferral and solver times of the gate (`clock_uq/gate_e2e.npz`).
- `gate_e2e.py` also reports the single-query cost ratio weighted by each
  halo's solver time (`speedup_single_query_weighted`); `energy_drift.py`
  records the maximum drift up to the marker (`max_drift_to_tx10`).

### Changed
- Recommended conductivity closure: alpha = 0.535, C = 0.886 (seven-halo fit)
  within 0.45 < Kn_c < 6.7; the five-halo fit (0.58, 0.85) is kept as
  `FIVE_HALO`. The harmonic closure of the public codes fails below
  Kn_c of about one. This supersedes the closure verdict and the reference
  uncertainties of `docs/INDEPENDENT_HALOES_AND_BARYONS_RESULTS_2026-11-07.md`
  (the earlier 6-8% resolution shifts of haloes B and C were seed scatter).
- The release accompanies the manuscript "GravOp: fast gravothermal evolution
  of SIDM haloes with a kinetically tested conductivity closure". The `statecond` benchmark remains in the release but is not part
  of that manuscript.

## [1.1.0] - 2026-11-06

### Added
- Disjoint-system transfer study of the deployment-penalty predictor: six
  equations outside the benchmark (quenching, porous-medium, complex
  Ginzburg-Landau, forced viscous Hamilton-Jacobi, and two adversarial
  variants), registered as a `statecond` plugin (`scripts/d1_systems.py`) and
  run and scored by `scripts/d1_campaign.py` under a protocol frozen before
  generation (`docs/DISJOINT_PREREGISTRATION.md`); results report
  `docs/TRACK_B_RESULTS_2026-11-06.md` and the `transfer` reproduction target.
- Per-termination bound on the SIDM label error, with a Richardson study of
  the Knudsen-terminated labels (`scripts/sc_clock_truth_sidm_mixed.py`,
  `scripts/sc_sidm_reference_knudsen.py`).
- Seed-disagreement construction of the displacement from validation data
  only (`scripts/e46_ensemble_displacement.py`).
- Conductivity-interpolation scan against the isolated N-body halo
  (`scripts/e2_conductivity_alpha/`, collected by
  `scripts/e2_conductivity_interpolation.py`).
- The emulator's training box in physical units for reference haloes
  (`scripts/emulator_training_box.py`) and the predicted-against-measured
  penalty figure (`scripts/fig_penalty_prediction.py`).
- Static baryonic potentials: `gravop.baryons` adds a Hernquist component to
  the hydrostatic equation (Zhong, Yang & Yu 2023), reproduces `gravop.solver`
  bit for bit without baryons, and is verified against published collapse
  times (`scripts/baryon_verification.py`). Controlled cases and a sensitivity
  grid on the calibration halo (`scripts/baryon_cases.py`), a baryonic dataset
  generator (`scripts/gen_baryon_dataset.py`) and a baryon-aware emulator with
  two further conditioning inputs (`scripts/train_baryon_emulator.py`); the
  `baryons` reproduction target and `docs/INDEPENDENT_HALOES_AND_BARYONS_RESULTS_2026-11-07.md`.
- Independent-halo validation against the public kinetic code NSphere: run
  definitions (`scripts/nsphere_reference_runs.py`), fluid, emulator and
  particle observables and their comparison (`scripts/independent_halo_analysis.py`),
  and the `independent_haloes` reproduction target.
- Selectable conductivity closure: `gravop.closure` takes the interpolation
  exponent alpha at run time (van den Bosch & Tokayer 2026), reproduces the
  production solver and the baryonic module bit for bit at alpha = 1, and
  records the legacy (alpha = 1, C = 0.712) and literature (alpha = 1/2,
  C = 0.9) closures. Three-halo transfer study (`scripts/closure_transfer.py`),
  the production dataset regenerated under another closure
  (`scripts/gen_dataset_closure.py`), the emulator re-aligned to the literature
  closure and evaluated (`scripts/closure_emulator_eval.py`), and the `closure`
  reproduction target; halo-B references at twice the particles (B5, B6).
- Figures 24 and 25 (`scripts/fig_baryons.py`, `scripts/fig_independent_haloes.py`).
- Three acceptance gates for the transfer study, five for the baryonic module
  (`tests/test_baryons.py`) and four for the closure module
  (`tests/test_closure.py`).

## [1.0.0] - 2026-09-26

First public release: this source tree and the archival record of results that
accompanies it.

### Added
- Gravothermal self-interacting dark matter fluid solver with semi-implicit
  conduction, for constant, power-law and velocity-dependent cross-sections,
  cross-validated against a public reference implementation.
- Four reference PDE systems whose evolution ends at a solution-dependent
  event time, with generators, event definitions and convergence studies:
  reaction-diffusion blow-up, Keller-Segel aggregation and thin-film rupture,
  each a finite-time singularity, and viscous Burgers steepening as a regular
  control.
- The rendered figures, in the archival record under `results/figures/`.
- Fourier neural operator emulator, physical-time forecaster, conformal
  calibration and uncertainty-gated execution.
- Bayesian inference pipeline for the cross-section parameters.
- The conditioning-coordinate study: oracle, deployed, state and physical-time
  arms across five systems and three operator backbones, the exact deployed
  error decomposition, and the solver-computable displacement predictor
  scored on reserved parameter draws.
- An acceptance suite of 229 gates that checks the reported results:
  158 verification gates, which recompute their assertion from the solver,
  the data or the model code, and 71 regression gates, which check a
  value against a stored record.
- Snapshot identity before a DOI exists: each tree carries a
  `MANIFEST.sha256`, whose SHA-256 names the tree, and
  `scripts/verify_release.sh` checks a tree against its manifest.
- `setup-env.sh`, a one-command environment bootstrap, and `requirements.lock`,
  the environment that produced the reported results.
- Community and publication metadata: contribution guide, code of conduct,
  licence decisions in `LICENSES.md`, `CITATION.cff`, `.zenodo.json`, issue
  and pull-request templates, and a continuous integration workflow.

### Not yet included
- The results of the continuous-prior population study, which is still
  running. Its frozen specification is in the archival record; its results
  will be a later release.

[1.0.0]: https://github.com/salahgo/gravop/releases/tag/v1.0.0
