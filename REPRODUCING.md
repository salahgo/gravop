# Reproducing the results

The release is two trees. This repository is the source release: code,
documentation and acceptance suite. The archival record on Zenodo holds
`results/`: every machine-readable result, and the rendered figures under
`results/figures/`. The suite needs both.

## Assembling the two trees

```bash
pip install -e ".[all]"
unzip /path/to/archival-record.zip 'results/*' 'experiments/*' 'data/*'   # at the root of this checkout
python scripts/reproduce.py verify --execute
```

`verify` runs the acceptance suite, which checks the reported results against
the solver, the data and the stored records. It needs no network, no GPU and
no external data. Tests
whose inputs are deliberately not archived — trained weights, the HDF5
trajectory ensembles, the externally supplied merger trees — skip and say so.

`reproduction.json` defines each target's commands, required inputs and
produced files. Omit `--execute` to inspect a plan; missing prerequisites and
failed commands return nonzero.

## What each target establishes

| Target | Scope |
|---|---|
| `verify` | The acceptance suite: in-process tests and stored-result consistency |
| `methods` | Canonical operator → converged clock → direct-time → measured-input comparison sequence, on the FNO backbone. The CNO and transformer sweep and the reserved-draw test are run by their own scripts (`scripts/sc_run_backbone.sh`, `scripts/e43_prospective_generate.py`, `scripts/e43_prospective_score.py`) |
| `forecast-results` | Audits of the application forecasting studies; needs the local datasets and checkpoints |
| `final-validation` | The frozen diagnostic prerequisites of the population study, failures retained |
| `fresh-confirmation` | The frozen continuous-prior population study; refuses to run until all six diagnostics pass |
| `transfer` | The disjoint-system transfer study of the penalty predictor (`docs/DISJOINT_PREREGISTRATION.md`): six new equations generated, trained under the unchanged recipe, decomposed and scored against the frozen criteria; retrains six operator suites and their clocks on one GPU |
| `baryons` | Verification of the static-baryon module against published collapse times, the controlled cases on the calibration halo, the 1200-trajectory baryonic dataset, three baryon-aware emulator members and their evaluation with the production ensemble; one GPU |
| `independent_haloes` | The NSphere reference runs for haloes B and C (built separately, see `DATA.md`), the fluid model at the halo-A calibration, the production emulator zero-shot and their comparison; the particle runs dominate the cost |
| `closure_test` | The conductivity-closure test of the manuscript from the archived particle measurements and fluid grids: ladders, scores of the pre-registered haloes D1, D2, E1, E2, the halo B half-step control, the five- and seven-halo fits, robustness, the matched-marker refit, the three closure-test figures and the closure tests; CPU, minutes |
| `uncertainty` | The clock-error propagation, coverage events, end-to-end gate and Figure of Section 6 from the archived baryon-aware members and dataset; one GPU, minutes |
| `closure_fit7_emulator` | The production dataset regenerated under the recommended seven-halo closure, five members retrained and evaluated; one GPU (the members are archived, so this target is needed only to rebuild them) |
| `closure` | The three-closure transfer study on haloes A, B and C, the production dataset regenerated under the literature closure, five emulator members retrained on it with the production recipe, and their evaluation; one GPU |
| `gpu-results` | Recomputes the evaluated GPU solver prototype's accuracy and throughput decision from local arrays; needs no GPU |

`methods` retrains and writes research outputs. Run it only in a separate
checkout prepared for rebuilding; it is not a read-only audit. Its clock sweep
reaches 20,000 epochs and saves the converged head. The standalone
`scripts/sc_experiment.py` keeps the 200-epoch-clock recipe because later
experiments depend on that version's identities; it is not the canonical final
comparison on its own. The application forecasting studies have frozen
protocols under `docs/FORECAST_*PROTOCOL.md`.

## A reproduction path that retrains nothing

Steps 1–4 read frozen weights and stored artefacts and together cover the
decomposition, the calibration analysis, the gate and the solver benchmark.
They need the trained weights, which are not archived. Step 5 needs nothing
but the two trees.

```bash
# 1. the decomposition, the metric bridge, and the displacement recomputed
#    from the validation split's clock errors (GPU ~2 min; the last on CPU)
python scripts/sc_decomposition.py  --device cuda:0
python scripts/sc_metric_bridge.py  --device cuda:0
python scripts/sc_deployable_displacement.py

# 2. the calibration, on both input paths    (GPU, ~10 s)
python scripts/sc_conformal.py          --device cuda:0
python scripts/sc_conformal_deployed.py --device cuda:0

# 3. the gate and its fallback cost          (GPU and stored wall times, ~30 s)
python scripts/sc_gated_deployment.py --device cuda:0
python scripts/sc_deferred_cost.py

# 4. the solver, against a closed form
python scripts/sc_exact_verification.py
#    the work-precision sweep takes ~25 min, mostly in the two explicit runs
#    that reach the 1500 s cap; its record is results/e8/workprecision.json
# python scripts/e8b_workprecision.py

# 5. the acceptance suite, against the stored results (CPU, ~4 min)
python -m pytest tests/ -q
```

Steps 1–4 write the artefacts step 5 reads, so step 5 alone checks the shipped
numbers and steps 1–5 check that the code still produces them.

Every table, and every figure drawn from stored summaries, is reproduced from
the archived results in minutes on one core (`scripts/make_figures.py`,
`scripts/sc_figures.py`). Four application figures — the solver cross-check,
the N-body comparison, the emulator examples and the posterior — also read
per-run arrays and emulator weights that the archive does not carry, and are
rebuilt with the expensive tier below.

## The conductivity-closure test

Everything in the closure test re-runs from the archive record without new
simulations. From the checkout root, with the record's `results/` and
`experiments/` unpacked:

```bash
python experiments/research_extensions/scripts/score_D.py           # frozen predictions against the particle runs
python experiments/research_extensions/scripts/closure_fit5.py      # five-halo fit, leave-one-out (seconds)
python experiments/research_extensions/scripts/closure_robustness.py
python experiments/research_extensions/scripts/score_E.py           # family E against its frozen predictions; halo B half time step
python experiments/research_extensions/scripts/closure_fit7.py      # seven-halo fit (post hoc), from the stored fluid grid
python experiments/research_extensions/scripts/marker_refit.py      # matched-marker refit (reads closure_scan_traj.npz; runs the D grid, minutes)
python scripts/fig_closure_test.py
python scripts/fig_closure_calibration.py
python scripts/fig_closure_structure.py                             # reruns 12 fluid trajectories (~1 min)
python -m pytest experiments/research_extensions/tests -q
```

The particle runs themselves need NSphere at commit `8f14666` (not
redistributed) and a few hundred thread-hours in total; each run's exact command, seed and
step count are in its `run.json`, and
`experiments/research_extensions/README.md` lists every stage. The
pre-registration and its timing are described in
`docs/CLOSURE_TEST_PREREGISTRATION.md`. The clock-error propagation study
(`experiments/research_extensions/scripts/clock_uq.py`,
`scripts/fig_clock_uq.py`), the end-to-end gate (`gate_e2e.py`) and the
coverage-event study (`clock_uq_events.py`) read the three baryon-aware
members and their dataset, both archived; on one GPU they replay the archived
records exactly, apart from the measured emulator time and the cost ratios
derived from it (`python scripts/reproduce.py uncertainty --execute`). `energy_drift.py` reads the raw NSphere energy dumps,
which are not archived; its record `closure_design/energy_drift.json` is.
The same sequence, with the figures and tests, is the `closure_test` target:
`python scripts/reproduce.py closure_test --execute`.

## Rebuilding from scratch

The five HDF5 benchmark sets under `data/sc/` are needed by the `methods`
target. Generate the four reference systems with
`python scripts/sc_gen.py --all --n 600`, and import SIDM with
`scripts/sc_import_sidm.py` after generating the application set as described
in [`DATA.md`](DATA.md). The application rebuild also needs the public
reference solver and the N-body data listed there. Regenerating the trajectory
ensembles and retraining the suite takes a few hundred core-hours and one
consumer GPU. `scripts/run_all.sh` is the historical application rebuild; it
requires `GRAVOP_REBUILD_HISTORICAL=1` and does not produce every current
artefact.

An FP64 GPU port of the solver was evaluated and not adopted: it matched the
CPU solver but was slower on the fixed 128-trajectory workload (48.3 s on 24
CPU workers against 125.3 s on two RTX 4090s). Its CUDA source and decision
record are `scripts/cuda/` and `results/gpu_solver/`; see
`docs/GPU_SOLVER_RESULTS_2026-09-11.md`.

`scripts/clean_env_smoke.sh` measures the from-scratch path on a fresh clone:
new environment, install, cheap data regeneration, gates. It accumulates
unexpected failures and returns nonzero.

## What is not archived, and why

- **Trained weights.** The application ensemble, the benchmark checkpoints
  and the forecasting models, about 2 GB in all. The exception is the
  five-member emulator retrained on the recommended seven-halo closure,
  which is archived (`results/closure_fit7_20261113/runs/*/model.pt`, about
  4.6 MB each) because it is the ensemble released for new work, and the
  three baryon-aware members with their 60 MB dataset
  (`results/baryons_20261107/runs/seed*/model.pt`, `data/baryons/train.h5`),
  which the uncertainty studies read. Each training run's full
  argument record and seed are archived (`results/runs/*/metrics.json`,
  `results/statecond/<system>_<architecture>.json`); the forecasting studies
  also record the SHA-256 of every model they load (for example
  `results/forecast_final/config.json`), and
  `results/statecond/ckpt_audit.json` records a behavioural check that each
  benchmark checkpoint is the architecture its name says. Every analysis that
  reads the weights writes its output into the archive, so the results are
  checkable without them.
- **Trajectory ensembles.** Regenerated from the released generators with the
  recorded seeds; the generation records are in the HDF5 attributes and in
  `data/sc/manifest.json`, which also records a content digest of each
  reference set (every dataset except the per-run wall times). Regenerate into
  a separate directory with `scripts/sc_gen.py --outdir DIR` and compare with
  `python scripts/sc_data_digest.py --check DIR`.
- **Intermediate solver state.** Per-run arrays that no table reads. Four
  per-record arrays are archived because a published claim is checked against
  them by hand: `results/statecond/conformal_deployed_records.npz` (coverage
  and gate decisions per trajectory), `results/e32/controls.npz` (the per-halo
  catalogue with its cosmological identities), `results/e31/trajectories.npz`
  (the matched pairs it was built from) and `results/e9/grid_zoom_extended.npz`
  (the solver-grid posterior of Figure 4, about nine core-hours to regenerate
  with `scripts/e9_zoom_grid.py`).
- **The population study's working state.** The fresh-confirmation run is
  still executing. Its frozen specification — configuration, generation
  record, overlap check, temporal truth — is archived; its per-case state is
  not, and will be deposited with the study's own report.
- **External data.** See [`DATA.md`](DATA.md) for sources, terms and checksums.

## Verifying a snapshot

Each tree carries a `MANIFEST.sha256` of every file under it, and the SHA-256
of that manifest is the tree's digest. To check a tree:

```bash
./scripts/verify_release.sh <tree>     # every file, then the tree digest
```

## Kinds of gate

Each test declares whether it is a `verification` gate, which recomputes its
assertion from the solver, the data, the model code or a checkpoint's own
behaviour, or a `regression` gate, which reads a stored result and checks a
quoted value against it. Select either with `-m verification` or
`-m regression`.

## Provenance limits

The application forecasting studies carry individual source, model and data
identities, and each later study verifies the identities of the earlier
sources and caches it reuses. `python scripts/sc_provenance.py` fails when a
required artefact or producer stamp is missing; `--legacy` opts into the older
permissive check. Three older benchmark artefacts carry no producer stamp. The
archive's manifest records their present contents; it cannot establish after
the fact which inputs produced them, and that limitation is stated rather than
repaired with invented provenance or a retraining it does not need.
