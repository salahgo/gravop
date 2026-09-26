# statecond — benchmark guide

`statecond` is a benchmark for surrogates of PDEs that become singular at a
solution-dependent time. It exists to answer one question under a fixed
protocol: **how much accuracy does a surrogate lose when the singular time it
was trained against has to be predicted rather than supplied?**

This document is for someone who wants to put *their own* architecture, or
*their own* equation, through that protocol and compare against our numbers.
The results themselves are in the archival record, under `results/statecond/`.

- Reference results, machine-readable: **`results/statecond/reference.json`**
  (regenerate with `scripts/sc_reference.py`)
- Reference results, human-readable: `results/statecond/tables.md`
  (regenerate with `scripts/sc_table.py`)
- Acceptance gates: `pytest tests/ -q`

---

## 1. The protocol

Three arms differing *only* in the scalar the operator is conditioned on.
Backbone, data, splits, seeds, optimiser, batch size and epoch budget are held
identical; changing anything else invalidates the comparison.

| arm | conditioning scalar | obtainable in use? |
|---|---|---|
| **A** `A_tau_oracle` | `tau = t / t_sing`, true `t_sing` supplied | no — this is the number papers report |
| **B** `B_tau_deployed` | `tau` from a *predicted* `t_sing` | yes — this is what the same model delivers |
| **C** `C_state` | `(phi, b)` read off the field | yes — needs no `t_sing` at all |
| **D** `D_time_direct` | standardised `log10 t`, no factorisation at all | yes — the user knows what time they asked about |

`phi = log10(A(t)/A(0))` for the diverging observable `A`, and `b` flags whether
the solution is past `argmin A`. The branch flag is not optional: `phi` is
non-monotone in most of these systems, so `phi` alone does not identify the
state.

Metric: field RMSE in units of the training-set field standard deviation
(nRMSE), which is defined for signed and log-transformed fields alike and so is
comparable across systems. Three seeds; the split is by trajectory (never by
time slice), 80/10/10, seed 1234.

**The endpoint policy.** Arm B's query `tau_hat = t / t_sing_hat` is clipped to
`[0, 1]`; the operator is never asked outside its training interval. Reproduce
this or your arm B is not comparable to ours — it is 7.7-18% of all deployed
queries in our five systems, and letting the query extrapolate instead is worse
by 1.1x to 10.3x. `scripts/sc_endpoint.py` measures both
(`results/statecond/endpoint.json`).

**Train the time head to convergence before quoting any penalty (P20).** Ours
was not: 200 epochs over trajectories is ~400 optimiser steps, and fixing that
alone moves the deployment penalty from 2.0-9.4x to 0.97-2.10x and arm C's
advantage from 2.2-35x to 1.07-7.46x. If you benchmark against arm B, report
your head's t_sing error and its convergence, or the comparison measures your
optimiser rather than your method.

**The baseline to beat is the best time arm in YOUR system, and it is not
always the same arm.** `sc_direct.py` measures arm D: it beats the *naive*
arm B in all five of our systems (3.2-6.7x), beats the E24 `tau`-augmented
model in all five, and in `burgers` beats arm C by 2.9x. But once the time head
is trained to convergence (P20), **arm B overtakes arm D in `rd`, `ks` and
`sidm`** — 0.0873 vs 0.0982, 0.0048 vs 0.0059, 0.0101 vs 0.0133 nRMSE — leaving
arm D strongest only in `film` and `burgers`. Measure both and quote your
advantage against the smaller; `results/statecond/contract.json` records the
winner per system in `best_time_source`. If your method only beats a naive
arm B, it has not yet beaten doing nothing clever.

**Headline quantities.** `B/A` is the deployment penalty. `B/C` is what state
conditioning buys. `sensitivity` is the error under a *10% imposed* `t_sing`
error divided by the oracle error — a property of the operator alone, with no
dependence on any time head.

---

## 2. Reproduce the reference numbers

```bash
pip install -e ".[all]"
python scripts/sc_gen.py --all --n 600        # the four systems, ~15 s
python scripts/sc_import_sidm.py              # the SIDM system (needs data/d1)
python scripts/sc_experiment.py --all         # three arms, FNO  (~20 min/GPU)
python scripts/sc_experiment.py --system sidm
python scripts/sc_table.py                    # markdown tables
python scripts/sc_law.py                      # the sensitivity law
python scripts/sc_direct.py --all             # arm D, the direct-time baseline
```

`--all` iterates the four bespoke PDEs only; `sidm` enters through
`sc_import_sidm.py` and must be named explicitly. `scripts/sc_run_backbone.sh`
does both plus the law fit for one backbone.

---

## 3. Adding your own architecture

An architecture is any `torch.nn.Module` that accepts the four sizing arguments
and maps an initial field plus a conditioning vector to a field:

```python
import torch.nn as nn
from statecond import register_backbone

class MyOperator(nn.Module):
    def __init__(self, n_fields=1, n_par=4, cond_extra=1, n_x=128, **kw):
        super().__init__()
        ...

    def forward(self, prof, cond):
        # prof: (B, n_fields, n_x)   cond: (B, n_par + cond_extra)
        # return:                    (B, n_fields, n_x)
        ...

register_backbone('mine', MyOperator)
```

`cond_extra` is 1 for the time-conditioned arms and 2 for the state-conditioned
arm; the same class must handle both, which it does automatically if it simply
consumes `cond` as a vector.

Save that as a module — say `myplugin.py` — and pass it with `--plugin`:

```bash
python scripts/sc_experiment.py --plugin myplugin.py --all --backbone mine --ckpt-dir
```

`--plugin` takes a dotted module name or a path to a `.py` file, and is
repeatable. It is imported *before* the argument parser is built, because
`--backbone` validates against the registry: registration happens in whichever
process performs it, so without this hook your name is simply rejected as an
invalid choice.

**`register_backbone` executes the contract rather than inspecting it**, and the
check includes a **backward pass** on both `n_fields = 1` and `2`. This is not
ceremony. On the machine this was developed on, Triton cannot JIT, so an op
whose backward is routed to a Triton kernel fails at runtime while every
forward-only smoke test passes — it cost a full training run to find, in a
contraction that was a batched outer product only when `n_fields == 1`. Pass
`validate=False` to skip the check if you have reason to.

**Fair-comparison note.** Report your parameter count. Ours are matched to
within 5% (FNO 587,457 · CNO 577,473 · transformer 615,545), deliberately, so
that "stronger" cannot be read as "bigger"; DeepONet at 232,448 is the
exception and is included as a deliberately weak point on the law.

---

## 4. Adding your own system

Subclass `System` and implement the contract:

```python
import numpy as np
from statecond import System, register_system

class MyPDE(System):
    name = 'mypde'
    n_fields = 1
    field_names = ('u',)
    param_names = ('a', 'b')
    n_solve = 512            # solver grid
    blowup_factor = 1e3      # A(t)/A(0) at which the run is declared singular

    def solve_grid(self): ...        # x on the solver grid
    def store_grid(self): ...        # x on the stored (128-point) grid
    def sample(self, rng): ...       # -> parameter vector
    def initial(self, p): ...        # -> solver state y0
    def rhs(self, t, y, p): ...      # -> dy/dt
    def observable(self, y): ...     # -> A(y) > 0, diverging at the singularity
    def store(self, y): ...          # -> (n_fields, n_x) stored fields

register_system(MyPDE)
```

`register_system` checks that every method is implemented, that `sample`
returns a **dict whose keys are exactly `param_names`** (`data.build` assembles
the parameter matrix as `[p[k] for k in param_names]`, so returning an array
fails later inside a worker process with an error that names neither the system
nor the cause), that `observable(initial(p))` is finite and strictly positive,
that `store` returns `(n_fields, n_x)`, and that `rhs` preserves the state
shape.

**`observable` is the one to get right.** The state coordinate is its
logarithmic contrast, so it must be positive and must *diverge* at the
singularity. If your singularity is a quantity going to zero — film rupture, for
example — return its reciprocal, as `Film` does. A sign or direction error here
does not raise; it silently produces a conditioning variable that means nothing.

Then generate and run:

```bash
python scripts/sc_gen.py --plugin myplugin.py --system mypde --n 600
python scripts/sc_experiment.py --plugin myplugin.py --system mypde
```

### `sidm` is an APPLICATION system, not a reference system

**Decided 2026-09-06.** The event rule above (`t_sing` is where
`A(t) = blowup_factor * A(0)`) defines the **four reference systems** — `rd`,
`ks`, `film`, `burgers` — and each honours it to 0.000 dex in *every*
trajectory (`results/statecond/event_spec.json`, gated by
`test_event_specification_holds_or_is_declared`).

**`sidm` is specified differently and is not a reference system.** It is
imported from the gravothermal training set rather than generated here, and its
event is:

> **solver termination, whichever of the density criterion
> (`rho[3] >= 1e3 rho[3](0)`) and the Knudsen fluid-validity floor
> (`Kn <= kn_min = 0.02`) fires first.**

Of the 603 retained runs (those where one of the two fires; the 697 where
neither does are dropped at import), **40% end on the density criterion and
60% on the Knudsen floor** — counted by `scripts/sc_event_spec.py`, not
asserted. Consequences you must know before comparing anything to `sidm`:

| | four reference systems | `sidm` |
|---|---|---|
| event | `A(t) = f A(0)` | solver termination, either criterion |
| `A(t_sing)/A(0)` | exactly `f` | median 225x, range 0.28-988x |
| `phi` endpoint | constant `log10 f` | spans **3.545 dex** |
| reproducible from this document? | yes | no — needs the gravothermal solver and its termination logic |

So `phi` is **not the same object** in `sidm`: there it carries per-trajectory
information about *where the run stopped*, which it cannot carry elsewhere.

**Two rules follow.** Do not calibrate a new system's numbers against `sidm`,
and do not read its convergence figures as a reference standard: `t_sing` is
evaluated at shell **index** 3, which encloses a different mass fraction on
every grid, so its `t_sing` carries a grid-dependent definition and a
convergence study on it would mix discretisation error with a change of
definition. That is a known and stated limitation of the application system,
not of the benchmark.

If you implement `blowup_factor` as documented, your system behaves like the
four reference systems. That is the intended behaviour, and it is what the
benchmark's numbers are comparable to.

### Ground truth for `t_sing` is a hard requirement

The whole benchmark measures a penalty caused by `t_sing` errors of a few
percent, so a new system's reference `t_sing` must be converged far better than
that. Check with `scripts/sc_convergence.py`, which integrates at half the production
resolution, at it, and at twice it, scores production against the finest, and
adds two grid-independent checks: a 100x tolerance tightening and a re-read of
`t_sing` at a third of the event threshold. `scripts/sc_reference_accuracy.py`
then puts all five systems on one footing and divides by the `t_sing` error the
time head actually makes, which is the number that matters
(`results/statecond/reference_accuracy.json`).

Ours land between 6e-07 and 1.3e-03 in median relative change, a margin of
**21.9x (`ks`) to 1.1e5x (`rd`)** against the *published* head's `t_sing` error.
Aim for at least the low end of that, and check the tail as well as the median:
on our worst single `ks` trajectory the margin is only 3.6x.

**Score against a Richardson limit, not against a 2x reference (P21).** We
refined the reference from 2x to 4x the production grid on a 24-trajectory
subset and Richardson-extrapolated it. The production error did not fall — it
*rose*, by 1.26-1.36x in all four reference systems (`rd` 2.81e-06 -> 3.70e-06,
`ks` 7.81e-04 -> 1.04e-03, `film` 1.55e-03 -> 1.96e-03, `burgers` 2.77e-04 ->
3.76e-04), because a 2x reference carries its own error in the same direction
and partially cancels. The observed order is 2.00, 1.99, 2.22, 2.00 on 24 of 24
trajectories each, so the schemes are behaving; **the limit is the PRODUCTION
grid and no reference refinement can move it.** A 2x-reference convergence study
will therefore report a margin that is too generous by about a third.

**The standard, and our own scores against it:** median margin >= 10 and >= 5 on
the difficulty-stratified subset, measured against the head you intend to
deploy and against a Richardson limit. On the converged head
(`results/statecond/refined_reference.json`):

| system | production grid | converged head `t_sing` err | margin (median) | margin (hard subset) | meets the standard |
|---|---|---|---|---|---|
| `rd` | 512 | 0.219% | **592.2** | **36.0** | yes |
| `film` | 256 | 0.822% | 4.2 | 1.5 | no |
| `burgers` | 1024 | 0.149% | 4.0 | 2.1 | no |
| `ks` | 384 | 0.037% | **0.4** | 0.1 | no |

So **one of our four reference systems meets the standard we ask of yours.** In
`ks` the grid's own error (0.104%) exceeds the head's (0.037%), which is why
the defensible statement there is that the head reproduces the benchmark's
*labels* to that accuracy, not the singular time. The measured penalties still
stand — head and targets share the same labels, and the residual correction is
at most 1.049x (`results/statecond/clock_truth.json`) — but a new system that
wants to make a converged-clock claim needs a grid that supports it.

**Those margins are quoted against the PUBLISHED time head (1.8-7.1%), and the
converged head (P20, 0.04-0.82%) does not clear them.** Recomputed against the
operating point the project now quotes, the same margins are `rd` 3406, `film`
10.1, `burgers` 5.0, **`ks` 0.44 and `sidm` 0.32** — below one, meaning the
reference `t_sing` is coarser than the head error being reported through it.
`sc_reference_accuracy.py` now emits both operating points and a verdict
(`_summary.converged`). **Set your target from the head you actually intend to
deploy, not from ours**: the requirement is median margin >= 10 and >= 5 on the
difficulty-stratified subset. A benchmark whose ruler is coarser than its
measurement will report a converged-clock result it cannot support.

**Check the difficult end, not just the median.** `sc_convergence.py --hard 8
--screen 48` screens candidates on the two cheap grids and carries the worst
through the full study, so the expensive reference is paid only for the ones
that matter. On ours the margin falls from 22x to **5.9x** (`ks`) and from 63x
to 15x (`film`). A median alone would have hidden that.

**We enforce this on ourselves, and we have now paid for it twice.** A 2-D
Keller–Segel testbed was built, worked, and was *withheld* because its `t_sing`
converged only to a few percent between the 128 and 256 grids — far worse than
the four production systems, and unable to support a study of a few-percent
effect. The cause was that the scheme was **first order in both operators**, so
it was rebuilt: MUSCL with a minmod limiter in space, SSP-RK2 in the
integrating-factor variable in time. That worked — **observed order 2.009 on
24 of 24 trajectories** — and it was **still not enough**
(`results/statecond/ks2d_convergence.json`):

| grid | rel. err vs Richardson limit | hard subset | margin vs a 0.822% head | 600 runs on 24 cores |
|---|---|---|---|---|
| 128 | 7.05e-02 | 1.94e-01 | 0.1 | 0.00 h |
| 512 | 2.30e-03 | 5.55e-03 | 3.6 | 0.44 h |
| 1024 | 1.06e-03 | 2.82e-03 | 7.8 | 6.89 h |

The standard is median >= 10 and hard-subset >= 5, so the system fails it even
at 1024, where the dataset alone costs 6.9 h; 2048 would pass at ~55 h. It stays
withheld. **The lesson for a new system is that in 2-D the grid, not the
scheme, is what you will be paying for** — and that a gentler event threshold
does not rescue it (`f = 10` instead of 30 is worth a uniform 1.7-1.9x, no
more). If a new system cannot meet the bar, its numbers are not comparable to
these, and reporting them anyway is worse than not adding it.

---

## 5. What to compare against

From `results/statecond/reference.json`:

- `points[]` — one entry per (system x backbone) with `A_tau_oracle`,
  `B_tau_deployed`, `C_state`, `deployment_penalty_B_over_A`,
  `state_vs_deployed_B_over_C`, `sensitivity_at_10pct`, and the test-set size
- `law` — `penalty ~ sensitivity^exponent` with its CI, the log-space
  correlation, and the within-system ordering test
- `protocol` — the conditions the numbers are valid under

And two artefacts that certify the reference rather than the results:
`results/statecond/reference_accuracy.json` (how well the reference `t_sing` is
converged, per system, against the error it must be small compared to) and
`results/statecond/endpoint.json` (the arm-B endpoint policy and what it costs).

If you add a backbone, `scripts/sc_law.py` will pick it up automatically and
place it on the same log-log fit, and will additionally report a **held-out**
prediction: the law fitted on the original two architectures only, then applied
to yours. That is the honest test, because your points did not inform the
coefficients.

### Conditions worth knowing before you quote a number

- **The effect is conditional on operator accuracy.** The penalty scales with
  phase sensitivity, so a weak operator shows a small penalty — and is not worth
  deploying. Measure `d(error)/d(t_sing error)` before concluding there is
  nothing to see.
- **State-conditioning needs the state coordinate to be measurable.** The
  Against the strongest time arm of each system (converged-clock B, or D where
  it is more accurate), the advantage reverses at a phase-measurement error of
  about 0.008 dex for `ks`, 0.020 for `film`, 0.043 for `sidm` and 0.058 for
  `rd`, interpolated linearly between the tested noise levels; in `burgers` D
  already beats noise-free state-conditioning. (Against the published-clock
  time arm trained with clock-error augmentation, the E24 comparator, the same
  curves cross at 0.138, 0.082, 0.042, 0.014 and beyond 0.2 dex.)
- **And measuring it costs more than that noise model suggests (P13,
  re-measured against the right comparator in P22 and with intervals in P23).**
  Reading `phi` off the field the operator consumes, through a 5% aperture, with
  `A(0)` known to 0.05 dex, and compared against the **best time arm** rather
  than against arm D. Ratios below are best-time / arm C, so > 1 means arm C
  wins; intervals are 95% paired trajectory bootstraps:

  | system | ratio at `sigma_a0` = 0.05 dex | 95% CI (oracle branch) | inferred branch | best time arm | reading | `A(0)` tolerance |
  |---|---|---|---|---|---|---|
  | `rd` | 0.920 | [0.718, 1.186] | 0.851 [0.675, 1.087] | B converged | **no resolved difference** | 0.043 dex |
  | `sidm` | 0.870 | [0.733, 1.048] | 0.871 [0.732, 1.037] | B converged | **no resolved difference** | 0.042 dex |
  | `film` | 0.439 | [0.327, 0.588] | 0.437 [0.326, 0.585] | D | arm C loses | 0.017 dex |
  | `ks` | 0.126 | [0.097, 0.171] | 0.126 [0.097, 0.170] | B converged | arm C loses | none exists |
  | `burgers` | 0.050 | [0.040, 0.062] | 0.050 [0.040, 0.061] | D | arm C loses | none exists |

  **Corrected 2026-09-12 (P22a).** The `film` and `burgers` rows previously read
  0.607 and 0.157. Those were arm B converged over arm C; the best time arm in
  those two systems is **D**, so they were the wrong comparator under a
  "best-time / arm C" heading. `sc_significance.py` now writes
  `bestTime_over_armC_full` and this table and its checker both read it.

  There is **no** `A(0)` tolerance at which arm C wins in `ks` or `burgers`;
  they fail on aperture and resolution alone. Run `scripts/sc_contract.py` on
  your own system before quoting an advantage. The branch flag, by contrast, is
  cheap: a small CNN on one field gets **94.4-99.8%** overall and 78.4-99.2% on
  the slowest quintile of trajectories, which is the population that matters.
- **The choice of state coordinate is worth up to 19x, and we cannot yet tell
  you how to make it (P24, P29a).** The same operator conditioned on a bulk L2
  contrast instead of the singular observable is 18.74x worse in `rd`, 3.81x in
  `ks` and 3.29x in `sidm`, and *better* in `film` (0.88x) and `burgers`
  (0.44x).
  A training-free criterion for that choice — how non-monotone the coordinate is
  along the trajectory — ordered all five systems perfectly under one statistic
  and **failed its pre-registered robustness test** under three others
  (`results/statecond/injectivity.json`). Choose the observable that tracks your
  singular structure, and *measure* the choice; do not assume any state
  functional will do.
- **It is a forward-prediction method.** Conditioning on state removes the
  dependence on `t_sing`, and in an inverse problem the parameter information
  lives there.
- **Quote both penalty ranges.** 2.0–9.4x is the naive recipe; 1.9–7.6x is after
  training the time-conditioned model against simulated `t_sing` error. Neither
  alone is the whole story.
