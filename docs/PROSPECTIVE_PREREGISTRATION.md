# Pre-registration — the displacement predictor, tested prospectively

Written 2026-09-18, **before any reserved case is generated, scored or even
solved**. Committed first.

## Why this exists

This is the one thing the paper cannot currently claim. The predictor

    p_hat = sqrt(E_A^2 + F^2) / E_A

carries no fitted coefficients, but it was written *after* the decomposition on
these five systems was in hand, its regime was assigned in hindsight, and its
aggregation was chosen because the identity closes under it. The manuscript says
so, in the abstract and in the subsection title ("stated and scored
retrospectively"), and a referee has accepted that wording **only because it
does not claim transfer**. Every other result in the paper rests on evidence;
this one rests on a disclaimer.

A disjoint, frozen, never-touched set of parameter draws now exists: 320
reserved confirmation cases, 64 per system, verified against F9's frozen 256 at
a minimum L1 separation of 4.5e-3 and lying strictly inside each system's
training prior. That is the missing ingredient, and this pre-registration fixes
what will be done with it before it is touched.

## What is fixed, now

**The rule.** `p_hat = sqrt(E_A^2 + F^2) / E_A`, unchanged, no refitting, no
coefficient. The bare rule `F / E_A` is scored alongside it because the paper
reports both, and the published ordering (quadrature better at both clocks) is a
prediction this test can falsify.

**The target.** `p = E_dep / E_A` per (system, backbone), scored as `log10 p`.

**The aggregation.** RMS of the log residual within a system, then the
unweighted mean over systems -- identical to Table 14. Because this paper has
been bitten once by an unstated aggregation, the pooled-RMS and per-trajectory
conventions of the metric bridge are reported alongside, but **the mean-of-system
RMS is the primary and is named now.**

**The nulls**, both of which must be run and reported whatever happens:
- a constant, fitted on the other four systems (the paper's `R0`);
- no penalty at all, `p_hat = 1` (the paper's `R_unit`).

**The operating point.** The converged clock. The published clock is reported as
a secondary because the paper reports both, but the converged clock is the one
the paper recommends and is the primary here.

**The sample.** All 320 reserved cases, 64 per system. If a system cannot be
generated for a stated technical reason, that reason is reported **and the
system is excluded before any scoring**, never after.

## The readings, fixed before the numbers exist

| outcome | reading |
|---|---|
| prospective score within ~1.5x of the retrospective 0.055 dex | the predictor transfers. The retrospective caveat is lifted, the abstract and subsection title are rewritten, and this becomes the paper's headline evidence |
| prospective score materially worse but still beating both nulls | the rule carries real information out of sample and the retrospective figure was optimistic. **The prospective number replaces it as the headline**, and the retrospective one is reported as the in-sample figure it is |
| prospective score fails to beat the constant null | the predictor does not transfer. It is reported as an in-sample diagnostic only, the title claim is narrowed, and this is stated as the paper's principal limitation |

**No fourth option, and no tuning.** The rule, the aggregation, the nulls, the
clock and the sample are all fixed above. If the result is bad it is reported
bad. The single most likely way to corrupt this test is to look at the answer
and then adjust an aggregation, drop a system or re-pick a clock, so each of
those is named here in advance.

## What makes this genuinely out of sample, and what does not

**Does:** the 320 draws were never used to train, tune, select, score or
diagnose anything in this project. No reference solve, model evaluation or
predictor score has touched them. They are disjoint from F9's reservations.

**Does not:** they are drawn from the *same* prior boxes as the training data,
so this measures transfer to fresh draws of the same families, not transfer to
new physics, new systems or new initial-condition families. The paper's existing
claims about initial-condition extrapolation are separate and are not touched.
The operators, the clock heads and the normalisations are the frozen ones; only
the evaluation data are new.

## Feasibility, established before committing

Trajectories are a deterministic function of their stored parameters --
`System.initial(p)` and `System.rhs(t, y, p)` consume `params` and nothing else
-- so no seed is required. Regenerating training trajectory `rd[0]` from its
stored parameters reproduces its `log10 t_sing` to **9.7e-9**, in 0.035 s.

Recorded generation costs put all 320 reference solves at **under two minutes of
CPU** (burgers 0.04 s, rd 0.07 s, film 0.15 s, ks 0.22 s, sidm ~1.3 s per
trajectory). Operator evaluation is GPU inference on frozen weights. Nothing is
trained.

## Risks stated in advance

1. **SIDM initial conditions.** The reserved records carry `ic` as a name
   (`nfw`, `cored`, `gnfw`) and an `rc`; the training set stores `ic_type` as
   an integer in {0, 1, 3}. The mapping must be established from the generator
   and verified by regenerating a training case, not guessed. If it cannot be
   established, SIDM is excluded with that reason recorded.
2. **The clock head on fresh draws.** Applying it to new parameters is exactly
   the deployment scenario, which is the point rather than a hazard -- but if
   its error on the reserved draws is far from its error on the test split, the
   two operating points are different and both are reported.
3. **Reserved draws are inside the prior but not identically distributed within
   it.** Any difference in coverage is reported, not corrected.
