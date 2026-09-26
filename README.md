# GravOp

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Conditioning coordinates for neural PDE surrogates — a solver-computable
predictor of deployed accuracy, with a fast gravothermal SIDM solver and
emulator as the reference application.**

A neural operator trained in normalised time `tau = t / t_sing` needs a
terminal clock that is available when the model is scored and unavailable when
it is used. This repository measures that cost on one protocol across five
systems, shows that the gap is organised by a displacement computable from the
solver alone before any network is trained, and applies the result to
gravothermal collapse of self-interacting dark matter haloes.

## Contents

The release is **two trees**, published separately.

| Tree | Where | Holds |
|---|---|---|
| source release | this repository | code, documentation, acceptance suite |
| archival record | a companion Zenodo record, linked here once it is published | `results/`: every machine-readable result, and the rendered figures |

Each tree carries a `MANIFEST.sha256`; the SHA-256 of that manifest names the
snapshot; `scripts/verify_release.sh` recomputes it.

| Path | Contents |
|---|---|
| `src/gravop/` | gravothermal solver, emulator, dataset builder |
| `src/statecond/` | the four reference PDE systems, operator backbones, data |
| `scripts/` | experiment, analysis and figure scripts; `scripts/cuda/halo.cu` is an evaluated GPU prototype that was not adopted |
| `tests/` | the acceptance-gate suite |
| `docs/` | pre-registrations, frozen protocols and the reports of the forecasting studies |
| `REPRODUCING.md` | what each reproduction target establishes and needs, a path that retrains nothing, and what is not archived |
| `DATA.md` | data provenance, URLs, licences, access conditions |
| `STATECOND_BENCHMARK.md` | the benchmark guide, gated against the records |
| `reproduction.json` | each target's commands, required inputs and produced files |

## Setup

```bash
git clone https://github.com/salahgo/gravop
cd gravop
./setup-env.sh
```

or by hand:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

Python 3.10 or later. The solver and the acceptance suite are CPU-only; a GPU
is needed only to retrain the emulator. `requirements.lock` records the
environment that produced the published numbers and is not an install script —
see its header.

## Usage

Evolve a halo:

```python
from gravop import solver

res = solver.run_halo(
    sigma_m=2.0, w=1.5,                      # dimensionless sigma_0/m and w
    model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
    n_shells=300, eps_t=3e-4,                # production settings
    timestep_mode='implicit', kn_min=0.02,
)
res['t'], res['r'], res['rho'], res['p']     # snapshot arrays
res['status']   # 0 collapse, 1 time horizon, 2 Knudsen floor
```

The emulator is used through the same package; `scripts/e11_emulator_vs_nbody.py`
is a complete worked example. Trained weights are not archived; they are
regenerated from the released generators with the recorded seeds and training
arguments.

Individual stages:

| Command | Produces |
|---|---|
| `python scripts/e1_verify.py conv` | solver convergence study |
| `python scripts/e1_crosscheck.py` | cross-validation against the public reference solver |
| `python scripts/e2_nbody.py profiles` | N-body validation |
| `python scripts/gen_dataset.py train 1300 --seed 20260827` | training set |
| `python scripts/train.py --model gravop --seed 0 --out results/runs/gravop_s0` | one ensemble member |
| `python scripts/evaluate_all.py` | accuracy, distribution shift, conservation, calibration |
| `python scripts/e8_bench.py` | runtime and work-precision benchmarks |
| `python scripts/e9_inference.py mock` | Bayesian inference |
| `python scripts/e10_resolution.py` | zero-shot resolution transfer |
| `python scripts/make_figures.py` | application figures |
| `python scripts/sc_figures.py` | method figures |

## Reproduction

Extract the archival record's `results/` at the root of this checkout
(`unzip <record>.zip 'results/*'`). The suite then runs end to end with no
network access, no GPU and no external data:

```bash
python -m pytest tests/ -q                  # the acceptance gates
```

Reproduction targets, from cheapest to most expensive:

```bash
python scripts/reproduce.py verify --execute   # gates and stored-result consistency
python scripts/reproduce.py methods            # canonical retraining sequence (plan)
```

Omit `--execute` to inspect a plan. `methods` retrains and writes research
outputs; run it only in a checkout prepared for rebuilding.
[`REPRODUCING.md`](REPRODUCING.md) gives the scope of each target and the
external data it needs.

Every table, and every figure drawn from stored summaries, is reproduced from
the deposited results in minutes on one core. Four application figures (solver
cross-check, N-body comparison, emulator examples, posterior) also read per-run
arrays and emulator weights the archive does not carry; they are rebuilt in the
expensive tier, which regenerates the trajectory ensembles and retrains the
suite: a few hundred core-hours and one consumer GPU.

To confirm which snapshot you hold:

```bash
./scripts/verify_release.sh <tree>     # every file, then the tree digest
```

## Validation

The acceptance suite checks the reported results against the stored records or
recomputes them from a source of truth. Each gate declares which:

```bash
python -m pytest tests/ -q -m verification  # recomputed from a source of truth
python -m pytest tests/ -q -m regression    # compared with a stored artefact
```

A verification gate computes its assertion in the session from the solver, the
data, the model code or a checkpoint's own behaviour, so a wrong number stored
under `results/` cannot make it pass. A regression gate reads a stored summary
or per-record array under `results/` and checks a quoted value against it or
recomputes one from it, so it pins consistency rather than correctness. Of the
228 gates, 158 are verification and 70 regression.

External validation:

- The velocity-dependence kernel reproduces the public GravothermalSIDM
  reference implementation bit-identically in double precision, and the full
  solver agrees with it to 0.13% in collapse time.
- Against the public isolated-halo N-body core-collapse run of Fischer, Yu &
  Dolag (2025), the solver agrees to 0.020 dex and the emulator alone to
  0.023 dex under a single fitted conductivity parameter.
- The exact error decomposition closes on every measured point to a relative
  residual below 1e-7.

If you are reviewing this work, start with
[`REPRODUCING.md`](REPRODUCING.md): it says how the two trees fit together,
gives a reproduction path that retrains nothing, and lists what is not
archived and why.

## Citation

Cite the software as:

```bibtex
@software{gravop,
  author  = {Gontara, Salah},
  title   = {GravOp: conditioning-coordinate benchmarks for neural PDE
             surrogates, with a gravothermal self-interacting dark matter
             solver and emulator},
  version = {1.0.0},
  url     = {https://github.com/salahgo/gravop}
}
```

[`CITATION.cff`](CITATION.cff) carries the same metadata in machine-readable
form; GitHub renders it as a **Cite this repository** button. Each release is
archived on Zenodo; its DOIs are added here and to `CITATION.cff` once they
are minted. Cite the concept DOI, which resolves to the latest version, when
the version does not matter, and the version DOI wherever a specific snapshot
is meant.

If you use the N-body benchmark, cite Fischer, Yu & Dolag (2025) as well; see
[`DATA.md`](DATA.md).

## Licence

Code is GPL-3.0-or-later ([`LICENSE`](LICENSE)); the copyleft term is
inherited from the GPLv3 reference implementation whose algorithmic lineage
the solver follows. Documentation and figures carry the terms recorded in
[`LICENSES.md`](LICENSES.md), which also states what is not redistributed
here. The gravothermal algorithm follows Balberg & Shapiro (2002), Nishikawa,
Boddy & Kaplinghat (2020) and Outmezguine et al. (2023).

## Contributing and contact

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Release history is in
[`CHANGELOG.md`](CHANGELOG.md).

Maintainer: Salah Gontara, Ecole Polytechnique de Sousse — salah.gontara@polytecsousse.tn.
Please open an issue rather than writing privately when the question is about
the code or a published number.
