# Track F — a solver-only gating signal for surrogate–solver fallback

**Pre-registered 2026-09-13, committed before the run.**

## Position against the literature

Lakshmanan and Chopra, *Hybrid Neural World Models* (arXiv:2605.28317, May 2026),
train a horizon-conditioned surrogate and recover a per-trajectory error map
**from the network's own forward passes**, then use it to gate a reference-solver
fallback: 26–72x CPU speedup in surrogate-only mode, and roughly half the
residual error when the gate is active. They report it competitive with deep
ensembles, learned error heads, gradient-magnitude indicators and locally
adaptive conformal prediction.

Every one of those signals, theirs included, requires a trained network.

This project has a candidate that does not. The coordinate displacement
`F_c(δ)` is computed from the reference solution alone and is available **before
any network is trained**, so it can inform the decision of whether to train a
surrogate for a given trajectory at all, not merely whether to trust one.

## The question

Does the solver-only displacement rank trajectories by deployed error as well as
a network-derived signal does?

## Design

Per system, on the frozen test split, with existing checkpoints and no retraining:

| signal | source | available before training |
|---|---|---|
| `F_c(δ̂_i)` | reference solution and the per-trajectory clock error | **yes** |
| ensemble spread | disagreement across the three trained seeds | no |
| oracle | the measured per-trajectory deployed error | not a signal |

Two measurements:

1. **Ranking.** Spearman correlation of each signal with the true per-trajectory
   deployed error.
2. **Gating.** Deferring the worst `q` fraction by each signal, the residual mean
   error over the retained trajectories, for `q` in {0, 0.1, 0.2, 0.3}. The
   oracle curve bounds what any signal can achieve.

## Predictions

**Primary.** The solver-only signal attains a Spearman correlation with the true
deployed error of at least 0.5 in at least 4 of 5 systems.

**Secondary.** At a 20% deferral rate the solver-only signal removes at least
70% of the residual-error reduction the oracle gate achieves, in at least 3 of 5
systems.

**Tertiary.** The solver-only signal is not worse than the ensemble spread by
more than 0.1 in Spearman correlation in at least 3 of 5 systems.

## Consequences, fixed in advance

- **Primary and secondary hold** → a solver-only gate is viable, and the
  distinguishing claim against the learned signals is supportable.
- **Primary fails** → the displacement does not rank individual trajectories
  even though it predicts system-level error. Report that gap plainly; it is a
  meaningful negative about the diagnostic's granularity.
- **Tertiary fails** → the ensemble spread is the better signal and should be
  recommended, with the solver-only signal reported as the cheaper alternative
  and its cost in accuracy stated.

No signal will be added after seeing results, and no deferral rate will be
chosen to favour a conclusion. This tests a ranking claim, not an ordering over
systems.
