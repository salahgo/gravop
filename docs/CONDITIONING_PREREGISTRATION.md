# Track D — coordinate conditioning, and the first-order law for displacement

**Pre-registered 2026-09-13, committed before the run.**

## Why

The referee made two corrections to the admissibility argument, both sharper
than what the manuscript claims.

**First**, injectivity is sufficient but not necessary. The condition that
matters for state reconstruction is

    c(t₁) = c(t₂)  ⟹  G(t₁) = G(t₂),

with the other inputs fixed: repeated coordinate values are harmless when the
states they index agree.

**Second**, injectivity does not imply usability. Where the inversion is
differentiable,

    dG_c/dc = ∂_t G / ċ ,

so a coordinate can be strictly injective and still badly conditioned wherever
`ċ → 0`. A binary branch flag addresses non-uniqueness and does nothing for
conditioning.

This separates four failure modes the manuscript currently conflates:
**uniqueness**, **conditioning**, **branch identification**, and **measurement
accessibility**.

## The testable consequence

The displacement of Equation (2) is, to first order in the coordinate error,

    F_c(δ) ≈ ‖dG_c/dc‖ · δ = κ_c · δ,

with `κ_c` the root-mean-square of `‖∂_t G‖ / |ċ|` over the trajectory and the
query distribution. This is a first-order law relating a quantity computed
pointwise on the reference solution to a displacement measured by resampling it.
It has not been checked, and it is falsifiable.

## Design

From reference solutions only, with no network:

1. Compute `κ_c` for `c ∈ {τ, φ, ψ}` in all five systems, with the branch flag
   applied where it exists.
2. Compare `F_c(δ)/δ` against `κ_c` at δ = 0.01, 0.02, 0.05, 0.10, 0.20.
3. Report the ambiguity fraction and the conditioning tail separately, so the
   two failure modes are never reported as one number.

## Predictions

**Primary.** At the smallest perturbation, δ = 0.01, the ratio
`F_c(δ) / (κ_c · δ)` lies in [0.8, 1.25] for at least 12 of the 15
(system × coordinate) points.

**Secondary.** That ratio decreases monotonically in δ in at least 12 of 15,
since higher-order terms add curvature the linearisation omits.

**Tertiary, and the referee's point stated as a measurement.** At least one
system is well conditioned in `κ` while being ambiguous in `c`, or the reverse.
If uniqueness and conditioning always agree, they are not separate failure modes
in this benchmark and should not be presented as such.

## Consequences, fixed in advance

- **Primary holds** → the displacement has a first-order law computable
  pointwise, which is a stronger and cheaper statement than resampling, and the
  conditioning number becomes the reportable quantity.
- **Primary fails** → the displacement is not first-order in the coordinate
  error at the perturbations used; report it as measured only, and state that
  the linearisation is inadequate.
- **Tertiary fails** → drop the four-mode taxonomy and report uniqueness alone.

No predictor will be selected after the fact. This tests one stated law, not a
search over candidate statistics; the ordering claims that failed at n = 5 in
earlier work are not revisited.
