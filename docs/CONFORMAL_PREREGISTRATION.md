# Pre-registration — P37: calibrated deployment guarantees for the emulator

Written 2026-09-14, before any number is computed. GPU only; F9 keeps the CPU.

## Why

Two independent pressures point at the same gap.

The referee wrote that a 25x increase in ensemble spread "is not itself a
validated failure detector", and that "coverage-tuned uncertainty inflation is
not independently validated coverage". Both are correct. The emulator's current
uncertainty is a deep-ensemble spread with no calibration guarantee, and the
inference pipeline inflates it using a factor selected on coverage results.

Two 2026 papers sharpen the point. Conformal prediction for neural operators
(arXiv:2606.09923) measures deep ensembles at **52.4% coverage against a 90%
target** on a heat-conduction operator, while split conformal with
uncertainty-normalised scores reaches 89.1%. And a closed-loop study of
surrogate gating (arXiv:2608.23075) shows that a *correct* out-of-distribution
gate can defer 96.8-99.7% of calls and turn a nominal speedup into a
0.94-0.96x slowdown, with a break-even deferral fraction

    d_break = (1 - g - r) / (1 - r),

`r` the surrogate-to-solver cost ratio and `g` the gate's own cost share.

## What is being built

Split conformal prediction over the existing 5-member ensemble. No retraining;
the published weights are untouched.

Score, per (trajectory, snapshot, radius) residual:

    s = |u - u_hat| / (sigma_ens + eps)

Threshold: the ceil((n+1)(1-alpha))/n empirical quantile of the calibration
scores. Interval: `u_hat +/- q * sigma_ens`.

Calibration set: the **validation** split. Evaluation: the **test** split.
Both are the frozen splits of the published training run, and neither was used
to fit the weights. Nominal levels 0.90 and 0.95.

## Pre-registered predictions

1. **The ensemble alone undercovers.** Treating `u_hat +/- z * sigma_ens` as a
   Gaussian interval at nominal 90% gives empirical coverage below 90% on the
   in-distribution test split.
2. **Conformal attains its nominal level in distribution**, within the
   finite-sample band `[1 - alpha - 1/(n+1), 1 - alpha + 1/(n+1)]`.
3. **Coverage degrades out of distribution.** Applying the same threshold to
   `ood_ic` and `ood_param` gives coverage below nominal.

## Fixed consequences

- Prediction 2 is a theorem for exchangeable data, not a hypothesis. **If it
  fails, the implementation is wrong and will be debugged, not reported as a
  finding.** That is the internal control on this experiment.
- If prediction 1 fails, the ensemble is already calibrated; conformal then
  adds a guarantee but no accuracy, and it will be reported that way rather
  than sold as an improvement.
- Prediction 3 is a measurement. Any outcome is reportable, including no
  degradation, which would itself be evidence the OOD sets are not far enough
  out to matter.
- Conformal's guarantee is void under distribution shift. That limitation is
  stated whatever the numbers do; it will not be presented as an OOD detector
  with a guarantee.

## The gating economics

Separately, and from measured costs rather than assumption: `r` from the
emulator and solver per-query timings already recorded, `g` from the measured
cost of evaluating the gate, and `d_break` from the formula above. Against it,
the deferral fraction a calibrated gate actually requires at a stated error
budget.

**Pre-registered consequence.** If the required deferral exceeds `d_break`,
gating cannot pay for this problem at this operating point, and that is
reported as a negative result rather than omitted. The existing Track F
correlation study does not answer this and is not a substitute for it.

## What this does not claim

Conformal coverage is marginal, not conditional: it holds on average over the
calibration distribution, not for every trajectory. No per-trajectory guarantee
is claimed, and no claim at all is made outside the training families.
