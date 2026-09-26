# Pre-registration — a conductivity *rate* from the halo104 merger trees

Written 2026-09-16, **before the estimator is built or run**. Committed first,
as with every pre-registration in this project, so that the reading below cannot
be chosen after seeing the answer.

## Why this exists

The paper has reported, since P28, that a second calibration of `C_eff` is not
obtainable from SIDM Concerto. V5 (P42) narrowed the reason from "no trees" to
"the release stores particle snapshots at one epoch, so an integration can only
start where NFW is still fair, and by then the halo is accretion-quiet". The
suite's authors have since pointed out (P50) that the main branch's
`(Vmax, R_Vmax)` evolution is itself informative about how the profile changes
with time. The trees in hand carry **236 epochs of exactly those two
quantities** for 81 matched halo pairs, so the obstacle is the *estimator*, not
the data.

E30 and E32 fit a **static phase** per halo: one number, at z = 0, testing the
*shape* of the track and explicitly not its rate. This estimator uses the time
axis the trees supply.

## The model being tested

The gravothermal fluid model in dimensionless variables depends on the halo only
through `(sigma_hat, w_hat)`; `gravop/units.py` states this and it is the reason
a table below is possible at all. For a halo whose structure evolves, the
gravothermal clock advances locally:

    dphi/dt = 1 / t_c(a; C_eff),     t_c = t_c_hat(sigma_hat, w_hat, C_eff) * t_0

with `(sigma_hat, w_hat, t_0)` evaluated at each epoch from the **CDM
counterpart's** structure at that epoch. The CDM run is the control: it is the
same halo without self-interaction, so it supplies the structure the halo would
have had, at every epoch, without the evolution being measured.

Integrating from the first resolved epoch to z = 0 gives a predicted phase

    phi_model(C_eff) = integral of dt / t_c(t; C_eff).

## The prediction

`phi_model(C_eff)` is compared against `phi_obs`, the per-halo phase already
fitted in E30/E32 and stored in `results/e32/controls.npz`. That fit is frozen:
it is not refitted here and does not depend on `C_eff`.

**Score:** RMS over haloes of `phi_model - phi_obs`, scanned over `C_eff`.

## What counts as success, fixed now

| outcome | reading, fixed in advance |
|---|---|
| the scan has an **interior** minimum, and the score there is materially below the score of the best constant-phase predictor | a rate calibration. Report `C_eff` and its uncertainty from the scan curvature |
| the scan is **monotone**, minimum at a grid edge | not a calibration. Report as a one-sided statement about the scanned range, with the same care P42 applies: an optimiser at a grid edge is not a confidence limit |
| the score never beats the **null** | the trees do not constrain the rate either; the P28 limitation survives its third test and is reported as such |

**The null**, which must be run and reported whatever happens: predicting
`phi_obs` by its own sample median, a one-parameter constant with no rate in it.
If the rate model cannot beat a constant, it has explained nothing. This is the
same nested-null trap that voided the original E30 structural claim, and it is
named here so it cannot be walked into twice.

**A second null**: `C_eff -> 0`, i.e. `phi_model = 0` for every halo. This is
"self-interaction does nothing", and it is *not* nested in the scan interior.

## Sample, fixed now

All 81 matched pairs, and separately the subset where the rate is measurable at
all. A halo whose `phi_obs` is 0 carries no rate information, and 23 of the 81
sit exactly at phase zero (E32). **Both** the full sample and the
`phi_obs > 0` subset are reported; the full sample is primary, because
restricting to moving haloes selects on the quantity being predicted.

## What could make this fail, stated before it does

1. **The integral's lower limit.** Starting at the first resolved tree epoch
   assumes no gravothermal evolution before it. If the halo was denser and
   hotter earlier, that assumption removes real signal and biases `C_eff` high.
   Reported as a sensitivity: the start epoch is varied and the answer re-read.
2. **Mergers and tides.** The fluid model has neither. Isolated haloes only
   (`pid == -1`), as in E30, and epochs after the last major merger are reported
   separately.
3. **The CDM counterpart is not the SIDM halo's own initial condition.** It is
   the same halo in a different run; the two branches share initial conditions
   but not their later mass accretion.
4. **Table interpolation error.** Measured against held-out direct solver runs
   and reported; if it exceeds 2% the table is refined rather than accepted.
5. **`t_c_hat` does not scale as `1/C_eff`.** Measured, not assumed: over
   `C in [0.4, 2.0]` the product `t_c_hat * C` drifts by 3% at
   `sigma_hat = 0.556` and by **35%** at `sigma_hat = 2.6`, so the table is
   built over `C` as a third axis rather than rescaled from one reference.

## What this is not

Not a second *independent* calibration in the sense P28 asked for, even if it
succeeds. It shares the fluid model, the solver and the Yukawa kernel with the
first, and the cross-section is the suite's rather than Fischer's. It tests
whether the rate the first calibration fixed is consistent with a cosmological
suite's time evolution. That is worth doing and it is a weaker claim, and the
paper will say so.
