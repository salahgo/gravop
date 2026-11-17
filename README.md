# GravOp

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**A fast gravothermal solver and neural emulator for self-interacting dark
matter (SIDM) haloes, with a conductivity closure tested against kinetic
particle simulations.**

The `gravop` package evolves the gravothermal collapse of SIDM haloes with a
semi-implicit Lagrangian solver, about three orders of magnitude cheaper than
explicit integration at equal accuracy. Its conductivity interpolation between
the short- and long-mean-free-path regimes is selectable at run time
(`gravop.closure`) and has been tested against NSphere particle simulations of
six haloes, four of them predicted before they were simulated, and against a
public N-body simulation. The harmonic interpolation of the public fluid codes
fails below a central Knudsen number of about one. The recommended closure
(`gravop.closure.CALIBRATED`, alpha = 0.535 and C = 0.886) fits all seven
references to within 5.5%, with a systematic uncertainty of about 5% in
collapse time. A Fourier neural operator emulates the density and
velocity-dispersion evolution with a learned collapse clock, and propagating
that clock's error gives uncertainty bands that decide when to call the
solver. Static baryonic potentials and Bayesian cross-section inference are
included.

The repository also holds `statecond`, a benchmark of conditioning coordinates
for neural surrogates of PDEs that end at a solution-dependent time (four
reference systems), which the SIDM emulator was originally developed against.
It is documented in [`STATECOND_BENCHMARK.md`](STATECOND_BENCHMARK.md).

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
| `src/gravop/` | gravothermal solver, its static-baryon extension (`baryons.py`) and selectable conductivity closure (`closure.py`), emulator, dataset builder |
| `src/statecond/` | the four reference PDE systems, operator backbones, data |
| `scripts/` | experiment, analysis and figure scripts; `scripts/cuda/halo.cu` is an evaluated GPU prototype that was not adopted |
| `tests/` | the acceptance-gate suite |
| `docs/` | pre-registrations, frozen protocols and results reports, including the disjoint-system transfer study (`docs/DISJOINT_PREREGISTRATION.md`, `docs/TRACK_B_RESULTS_2026-11-06.md`) |
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
environment that produced the published numbers and is not an install script;
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

With the closure calibrated against the particle references:

```python
from gravop import closure, solver

r, m, rho, p = solver.nfw_profile(0.01, 100.0, 300)
res = closure.run_halo_g(sigma_m=2.0, **closure.CALIBRATED, r=r, m=m, rho=rho, p=p,
                         eps_t=3e-4, timestep_mode='implicit', kn_min=0.02)
```

The emulator is used through the same package; `scripts/e11_emulator_vs_nbody.py`
is a complete worked example. The five members retrained on the recommended
closure are in the archival record
(`results/closure_fit7_20261113/runs/gravop_fit7_s*/model.pt`) and load with
`load_model` in `scripts/evaluate_all.py`; so are the three baryon-aware
members (`results/baryons_20261107/runs/seed*/model.pt`). The other trained
weights are not archived; they are regenerated from the released generators with the recorded
seeds and training arguments.

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
| `python experiments/research_extensions/scripts/closure_fit5.py` | five-halo closure fit and leave-one-out (needs the archive record) |
| `python experiments/research_extensions/scripts/score_E.py`, `closure_fit7.py` | test on the pre-registered family E, seven-halo fit (recommended closure) |
| `python scripts/reproduce.py closure_test --execute` | every stored-data stage of the closure test, its figures and tests (CPU, minutes) |
| `python scripts/fig_closure_test.py`, `fig_closure_calibration.py`, `fig_closure_structure.py` | closure-test figures |
| `python scripts/fig_clock_uq.py` | clock-error propagation figure (GPU) |
| `python scripts/sc_figures.py` | method figures |

## Reproduction

Extract the archival record's `results/`, `experiments/` and `data/` at the
root of this checkout (`unzip <record>.zip 'results/*' 'experiments/*' 'data/*'`). The suite then runs end to end with no
network access, no GPU and no external data:

```bash
python -m pytest tests/ -q                  # the acceptance gates
```

Reproduction targets, from cheapest to most expensive:

```bash
python scripts/reproduce.py verify --execute   # gates and stored-result consistency
python scripts/reproduce.py closure_test --execute  # the closure test from the archived measurements
python scripts/reproduce.py uncertainty --execute   # clock-error, coverage and gate studies (one GPU, minutes)
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
recomputes one from it, so it pins consistency and not correctness. Of the
241 gates, 168 are verification and 73 regression.

External validation:

- The solver agrees with the public GravothermalSIDM reference implementation
  to 0.13% in collapse time, and its velocity-dependence kernel is identical in
  double precision.
- Against the public isolated-halo N-body core-collapse run of Fischer, Yu &
  Dolag (2025), the solver agrees to 0.020 dex and the emulator alone to
  0.023 dex under a single calibrated conductivity.
- Against NSphere particle references converged in particle number and time
  step (haloes B, C and the pre-registered D1, D2), the harmonic closure misses
  the collapse time by up to 20% (5.9 sigma on D1, chi2 57 over five haloes);
  the five-halo fit (alpha 0.58, C 0.85) has chi2 3.9 for 3 degrees of freedom
  (4.2 from direct solver runs at the best fit) and predicts every halo left out of the fit to within 7%. Frozen and tested
  on a further pre-registered family, it passed halo E1 inside its range and
  failed E2 below it (11% early); the harmonic closure misses E2 by 29%. The
  fit to all seven haloes (alpha 0.535, C 0.886; chi2 10.0 for 5 degrees of
  freedom) is the recommended closure. It was fitted after that test and has
  not itself been tested on new haloes. Code and protocol:
  `experiments/research_extensions/README.md`,
  [`docs/CLOSURE_TEST_PREREGISTRATION.md`](docs/CLOSURE_TEST_PREREGISTRATION.md),
  [`docs/CLOSURE_TEST_RESULTS_2026-11-12.md`](docs/CLOSURE_TEST_RESULTS_2026-11-12.md).
- Propagating the emulator's clock error raises the AUROC for detecting
  predictions with more than 10% error from 0.80 to 0.95 on test haloes.
- The static-baryon module reproduces the published fluid collapse times of
  Zhong, Yang & Yu (2023) to within 2% for their constant cross-section of
  10 cm^2/g, 5% for their most compact baryons and 12% at 100 cm^2/g.

If you are reviewing this work, start with
[`REPRODUCING.md`](REPRODUCING.md): it says how the two trees fit together,
gives a reproduction path that retrains nothing, and lists what is not
archived and why.

## Citation

Cite the software as:

```bibtex
@software{gravop,
  author  = {Gontara, Salah},
  title   = {GravOp: fast gravothermal evolution of self-interacting dark
             matter haloes, with a kinetically tested conductivity closure},
  version = {1.2.0},
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

Maintainer: Salah Gontara, Ecole Polytechnique de Sousse, salah.gontara@polytecsousse.tn.
Please open an issue rather than writing privately when the question is about
the code or a published number.
