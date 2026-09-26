# Track A — the error decomposition, with its cross term

**Pre-registered 2026-09-13, committed before the run.**

## Why

The referee showed that the displacement is not a lower bound on deployed error.
With a learned operator written as \(\hat G = G + e\),

    Ĝ(ṽ) − G(v) = [G(ṽ) − G(v)] + e(ṽ) = Δ + e(ṽ),

so the deployed squared error is

    E_dep² = ‖Δ‖² + ‖e(ṽ)‖² + 2⟨Δ, e(ṽ)⟩.

The quadrature this project has used, `E_dep ≈ sqrt(E_A² + F²)`, makes two
substitutions it never tested: it discards the cross term, and it replaces
`‖e(ṽ)‖`, the operator error at the **corrupted** coordinate, with `E_A`, the
error at the **correct** one. Both are assumptions, not consequences.

This measures all four quantities directly. No retraining: 225 checkpoints
exist, including the 15 converged clock heads.

## Design

For every system, backbone and seed, on the frozen test split, in one
consistent inner product and one averaging convention:

| symbol | definition |
|---|---|
| `Δ` | `G(ṽ) − G(v)`, from reference solutions only |
| `e(ṽ)` | `Ĝ(ṽ) − G(ṽ)`, operator error at the deployed coordinate |
| `e(v)` | `Ĝ(v) − G(v)`, operator error at the correct coordinate |
| `χ` | `2⟨Δ, e(ṽ)⟩ / E_dep²`, the discarded cross term as a fraction |
| `ρ` | `‖e(ṽ)‖ / ‖e(v)‖`, the substitution the quadrature makes silently |

Both clock conditions are evaluated: the original head and the converged head.

## Predictions

**Primary.** At the converged clock, over the 25 (system × backbone) points,
`|χ| < 0.10` in at least 20.

**Secondary.** `ρ ∈ [0.8, 1.25]` in at least 20 of 25.

## Consequences, fixed in advance

- **Both hold** → the quadrature is an empirical approximation with measured
  validity conditions. Report it as such, with the residual distribution and the
  points where it fails, and retain it as a diagnostic.
- **Primary fails** → the cross term is not negligible. **Retire the quadrature**
  and report the exact decomposition instead. The deployment audit does not
  depend on it.
- **Secondary fails** → the substitution of `E_A` for `‖e(ṽ)‖` is unsafe; report
  `‖e(ṽ)‖` explicitly wherever the quadrature currently appears.
- **`χ` is systematically negative** → deployed error lies *below* the
  displacement, directly demonstrating the cancellation the referee described.
  This must be stated plainly as a confirmation of the criticism.

No coefficient will be fitted to close any gap, and no point will be excluded
after the fact. The outcome is reported whichever way it falls.
