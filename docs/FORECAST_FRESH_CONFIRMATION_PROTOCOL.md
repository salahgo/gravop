# Prospective fresh confirmation — preflight correction and fixed decision rules

This addendum applies only after every F8 diagnostic prerequisite passes.
It does not modify the active F8 diagnostic sources, gates, selected cases or
budget. No fresh confirmation trajectories have been generated under either
seed at the time of this correction (11 September 2026).

The F8 protocol inadvertently proposed parameter seed 980000000, which was
already used by F4. A parameter-only preflight reproduced 256/256 F4 draws.
The prospective parameter seed is corrected to **990000000**, with independent
observation-noise seed **990000001**. Parameter-only overlap checks must pass
at exact and float32 model-input precision against D1, F3 calibration and
confirmation, F4 confirmation and all F5/F6/F7/F8 solver nodes before any
fresh physics runs. Seed replacement is a preflight identity correction, not
selection on predictions, coverage or outcomes. Retain the original F8 protocol
and the recorded failed overlap result.

Generate exactly 256 uniform-log-prior trajectories. Retain every numerical
failure and all survival-conditioned observations at times 100/400/900. Check
every generating truth at the stricter level-2 setting before posterior analysis;
require unchanged support and maximum common profile difference <=0.00434 dex.
No selection on successful inference and no neural/discrepancy tuning.

Use the unchanged generalized F8 conditional integration, per-observation budget
80,000 submissions / 1,024 per column / 256 inner panels, and all original
0.02/0.005/(conditional0.00125), mean/width/CDF and five-peak temporal gates.
Resume exact cached results. Any required numerical failure stops confirmation
and blocks a population claim; partial passing cases are not a calibration set.

Only when every observation passes, summarize solver/candidate posterior CDFs
of both parameters and the solver likelihood. Across nine tests per method,
require Holm-adjusted uniformity p>=0.05, parameter 90% central-CDF coverage
between 0.85 and 0.95, and candidate median TV<=0.10 at every time, matching F3's
original screening thresholds. Report 68/90/95% coverage and Wilson intervals,
support errors and all failures. Solver controls must pass. The data-ignoring
control must fail the likelihood test; diagnose its numerical prior approximation
if it does not. Passing finite tests is evidence within this observational
model, not a proof of universal calibration or automatic production acceptance.

Before execution freeze this addendum, implementation, models, inherited
inputs, overlap report and case-generation configuration in a separate fresh
confirmation manifest. F8 success alone does not finish this conditional stage.

The data-ignoring control uses one independent prior draw per observation,
conditioned only on survival at that observation time (parameter stream
991000000 and rank stream 991000001, each indexed by observation). It runs only
after all posterior integrations pass. For each quantity, report the randomized
rank `(I(draw < truth) + U*(1 + I(draw == truth)))/2`. Under exchangeability
this is uniform even with a single draw; ties are explicitly randomized. Using
an independent draw for each observation avoids shared empirical-prior-bank
dependence. Holm adjustment covers its nine tests; require rejection of the
solver-likelihood quantity at every time. One draw limits power and is frozen
before outcomes; do not increase its size after seeing a failure. Record all
rejection-sampling trajectories and numerical failures. The control's level-1
forward model is exactly the one used to generate its reference likelihoods.

Before fresh generation, a separate GPU performance study was run.
Its prototype passed CPU agreement but failed the fixed throughput screen and
was not adopted. Include `results/gpu_solver/development_params.npz` in both
the exact/float32 overlap check and the fresh manifest. The fresh experiment
retains the same CPU solver, parameters/noise seeds and all original gates.

---

## Reporting amendment — 11 September 2026

**This amendment changes the stopping rule only. It changes no seed, no model,
no gate threshold, no budget and no numerical method.** The per-observation
integration, its 80,000/1,024 budget and all 0.02/0.005/0.00125, mean/width/CDF
and five-peak temporal gates are exactly as frozen above.

### What changed

The frozen rule was:

> Any required numerical failure stops confirmation and blocks a population
> claim; partial passing cases are not a calibration set.

That makes the experiment a conjunction of 563 gates whose per-case pass rate is
evidenced by 6 diagnostic observations. At the measured rate of 2.78 h/case on
24 logical CPUs the sweep runs ~65 days, and its single most likely outcome is
one failure partway through, yielding **no reportable quantity at all**.

The replacement rule is:

> Attempt every observation. Record each one as converged or failed, and report
> the convergence rate with a Wilson 95% interval. Compute the calibration
> screens on the converged subset and label them as conditional.

### Why this is not selection on outcomes

The amendment was decided on **feasibility** — the wall-clock projection and the
mismatch between a 563-fold conjunction and a 6-case evidence base — and not on
any result. At the moment of the decision **3 of 3 attempted observations had
passed and the fourth was still integrating**. No numerical failure had been
observed, so no failure was being rescued. The pre-amendment manifest is
preserved at `results/forecast_fresh_confirmation/config_before_reporting_amendment.json`
and the original rule is retained verbatim above.

### What the amendment must not do, and does not

Screens computed on the converged subset are **conditional on numerical
success**, which is a selection: an observation whose integration fails its
gates is plausibly one whose posterior is hardest, so dropping it can bias
coverage optimistically. The amendment therefore reports two distinct flags and
never merges them:

| field | meaning |
|---|---|
| `population_calibration_passed` | the **frozen** claim — screens pass **and** zero numerical failures |
| `population_calibration_passed_conditional` | the weaker claim — screens pass on the converged subset only |
| `numerical.failure_rate`, `numerical.failure_rate_ci95` | the new primary quantity |
| `selection_caveat` | states the bias in the artefact itself |

The negative control is restricted to the same converged subset, so the control
and the method screens are evaluated on identical observations. Each control
draw keeps its original observation-indexed seed, so restricting the set does
not reseed any draw.

Gated by `test_reporting_amendment_reports_the_failure_rate_instead_of_stopping`
and `test_the_strict_population_claim_still_requires_zero_numerical_failures`.

---

*Publication note.* Sentences recording how this study was authorised and scheduled have been reworded for publication; nothing the protocol commits to has changed. The experiment records name the frozen text by its SHA-256, `5b20f274fa44515a1d3a7090a8b2137686edc94691d6602b710311889ad42631`.
