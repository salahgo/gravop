# P32 — the branch flag, tested causally

**Pre-registered 2026-09-12, committed before any training run.**

## Why

P31 established a structural fact from reference solutions alone: the ambiguity
barrier on arm C's coordinate is negligible **with** the branch flag
(`B_phi` = 1.07e-05 to 7.03e-05) and large **without** it. That is an
observation about the coordinate, not about a network. This tests it causally.

The project's risk register names the alternative explanation directly: *"the
flag is capacity, not injectivity."* The design below holds capacity fixed.

## Design

Train arm C with the flag channel **zeroed**, not removed. The network keeps
`cond_extra = 2` and identical parameter count, optimiser, schedule, data,
splits, seeds and budget; only the *information* in that channel is destroyed.
Any degradation is therefore attributable to the branch information and not to
capacity.

`scripts/sc_experiment.py` and `src/statecond/model.py` are **pinned by the
running F9 calibration** (verified in both `config.json` manifests), so neither
is modified. The ablation zeroes `d['flag']` in memory before `tensors()`.

## The predictor, computed before training

`B_phi` from `scripts/sc_ambiguity.py`, with and without flag separation:

| system | `B_phi` with flag | `B_phi` no flag | ambiguous frac, no flag | `E_C` | predicted no-flag error | predicted ratio |
|---|---|---|---|---|---|---|
| `sidm` | 1.07e-05 | 0.118395 | 0.609 | 0.0043 | 0.1185 | **27.6x** |
| `rd` | 6.99e-05 | 0.127137 | 0.365 | 0.0117 | 0.1276 | **10.9x** |
| `burgers` | 4.46e-05 | 0.004665 | 0.063 | 0.0051 | 0.0069 | 1.36x |
| `film` | 7.03e-05 | 0.000850 | 0.012 | 0.0214 | 0.0214 | 1.00x |
| `ks` | 1.39e-05 | 0.000025 | 0.002 | 0.0030 | 0.0030 | 1.00x |

Predicted no-flag error is `sqrt(E_C^2 + B_noflag^2)`.

**Note on why this quadrature is better posed than P31's.** P31 used it across
*different* arms and it failed, because arm C's error is not a floor for a
separately trained arm E. Here both sides are the *same* arm under the same
recipe, so `E_C` is the right baseline by construction.

## Predictions

**Primary.** The degradation ordering matches the `B_phi` no-flag ordering:
`sidm` and `rd` degrade most, `ks` and `film` least. At n = 5 the exact
two-sided permutation p floor is **0.0167**, and that will be quoted, never an
asymptotic value.

**Secondary, quantitative.** The predicted ratio is within a factor of 2 of the
measured ratio in **at least 4 of 5** systems.

## Consequences, fixed in advance

- **Both hold** → the flag's role is established causally, and P31's structural
  result is promoted from an observation about coordinates to a property of the
  trained arm.
- **Primary holds, secondary fails** → report the ordering and the mechanism;
  do not report the quadrature as a predictor.
- **`ks` or `film` degrades strongly** → the flag is capacity or something else,
  not injectivity. **Retract the structural claim explicitly**, as was done for
  spectral bias and for N10. Do not reinterpret.
- **Nothing degrades anywhere** → the barrier does not reach the trained model;
  P31's structural half is a statement about reference solutions only and must
  be scoped to that in the manuscript.

No fitted coefficient will be introduced to rescue any outcome.
