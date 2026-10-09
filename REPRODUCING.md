# Reproducing the results

The release is two trees. This repository is the source release: code,
documentation and acceptance suite. The archival record on Zenodo holds
`results/`: every machine-readable result, and the rendered figures under
`results/figures/`. The suite needs both.

## Assembling the two trees

```bash
pip install -e ".[all]"
unzip /path/to/archival-record.zip 'results/*'   # results/ at the root of this checkout
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
  and the forecasting models, about 2 GB in all. Each training run's full
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
  `data/sc/manifest.json`.
- **Intermediate solver state.** Per-run arrays that no table reads. Three
  per-record arrays are archived because a published claim is checked against
  them by hand: `results/statecond/conformal_deployed_records.npz` (coverage
  and gate decisions per trajectory), `results/e32/controls.npz` (the per-halo
  catalogue with its cosmological identities) and `results/e31/trajectories.npz`
  (the matched pairs it was built from).
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
