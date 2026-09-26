# Pre-registration — the deployed-accuracy predictor, specified and tested

Written 2026-09-14, before the converged-clock evaluation is run. Referee
comment: the title claims a predictor of deployed accuracy, and a predictor is
a claim only if the rule, its inputs and its applicability regime are stated
and it is tested out of sample.

## What is being predicted

The **deployment penalty** of one (system, backbone) point,

    p = E_dep / E_A,

scored as `log10 p`. This is a ratio, not an absolute error. Predicting the
absolute deployed error additionally requires `E_A`, which is stated in the
information budget below rather than hidden in it.

## The candidate rules, with their information budgets

| rule | formula | inputs | fitted coefficients | operator trained on the target system | imposed-error sweep |
|---|---|---|---|---|---|
| R0 null | `log p` = mean over the fitted systems | penalties of other systems | 1 | no | no |
| R1 quadrature | `p = sqrt(E_A^2 + F^2) / E_A` | `E_A`, `F` | 0 | **yes**, oracle evaluation only | no |
| R2 bare | `p = F / E_A` | `E_A`, `F` | 0 | **yes**, oracle evaluation only | no |
| R3 power law | `log p = a + b log s` | `E_A`, `s` | 2 | **yes**, plus the sweep | **yes** |

`F` is the coordinate displacement, computed from reference solutions with no
network. `E_A` is the operator's error at the exact coordinate. `s` is the
phase sensitivity, the error under an imposed 10% terminal-time error divided
by `E_A`.

**R1 and R2 are not network-free.** Both consume `E_A`. Their saving against R3
is that they need no imposed-error sweep and fit nothing. The manuscript
currently states that the displacement predictor requires "no operator already
trained on the system"; if that reading is not supported here it will be
corrected.

## Protocol

Leave-one-system-out. R0 and R3 are fitted on the four retained systems and
scored on the held-out one. R1 and R2 fit nothing, so every point is out of
sample by construction and they are scored identically for comparability.

Aggregation, stated because the decomposition does not close under the other
convention: RMS of the `log10` residual **within** the held-out system, then the
unweighted mean across the five held-out systems. The pooled RMS over all
points is reported alongside it.

Two operating points, reported separately and never mixed:

- **published clock** — 25 points, the head of Section 3.2;
- **converged clock** — 15 points, the head of Section 4.6.

`s` was never measured at the converged clock, so R3 is scored at the published
clock only. That is a limit of the evidence and is reported as one.

## Pre-registered consequences

1. **R1 beats R3 at the published clock and beats R0 at the converged clock.**
   The predictor claim stands, scoped to the ratio and to the stated inputs.
2. **R1 beats R3 at the published clock but not R0 at the converged clock.**
   The claim is retired at the operating point the paper recommends, and the
   title drops "predictor of deployed accuracy". The displacement is reported
   as a diagnostic that orders systems, not as a predictor.
3. **R1 does not beat R0 at either clock.** The predictor claim is withdrawn
   entirely and the referee's suggested title is adopted.

A rule that loses to a constant is not a predictor. No coefficient will be
fitted to rescue R1, and no rule will be added after the numbers are seen.

## Why the cross term makes this a real test

Section 4.6 already reports that at the converged clock the quadrature
over-predicts by a median factor 1.10, because the cross term is negative in 13
of 15 points. R1 is therefore expected to degrade. The question this settles is
whether it degrades past the point of usefulness.
