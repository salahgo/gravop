# F10b — multi-seed replication of the factorial

**Pre-registered 2026-09-12, committed before any training run.**

## Why

F10 ran 12 validation-only fits at **seed 0 only** and was labelled a pilot in
its own artefact. Its headline differences are small -- full versus
features-only is 5.5% for SIDM and **-0.23%** for RD -- and a single seed cannot
separate effects that size from initialisation noise. Either the effects
replicate or the factorial does not enter the manuscript.

## Design

A new output directory `results/forecast/factorial_v2` with its own manifest
recording `seeds = [0, 1, 2]`. F10's `factorial_v1` is **not modified**: it is
committed, referenced by `docs/FORECAST_FACTORIAL_RESULTS_2026-09-11.md`, and
stays as the pilot record.

Seed 0 is re-run rather than copied. That costs one third of the compute and
buys a **reproducibility check against v1**: the same code, data and seed must
reproduce v1's validation nRMSE. A mismatch there invalidates the replication
before any seed effect is interpreted.

36 fits total (3 systems x 4 arms x 3 seeds), validation splits only, no test
evaluation, no fresh-confirmation claim. GPU only; F9 owns the CPU.

## Predictions

**Primary.** The sign of the clock-feature benefit is stable across all three
seeds for **SIDM** and **RD**: `full` beats `direct` at every seed in both.

**Secondary.** **Burgers** continues to favour the direct hybrid arm at every
seed.

**Reproducibility.** Seed 0 in v2 reproduces seed 0 in v1 to within 1e-6
relative on validation nRMSE for all 12 fits.

## Consequences, fixed in advance

- **Primary and secondary hold** → the factorial is replicated and may be
  reported, with per-seed spread shown, as a development result. It still makes
  no fresh-test or population-calibration claim.
- **A sign flips at any seed for SIDM or RD** → the pilot measured
  initialisation noise. **The factorial does not enter the manuscript**, and
  F10's result is recorded as not replicated.
- **Burgers flips** → report it; the direct-hybrid preference was a one-seed
  artefact.
- **Seed 0 fails to reproduce v1** → stop and diagnose. Do not interpret any
  seed effect until the discrepancy is explained.

RD's -0.23% is already inside plausible seed noise, so the honest expectation is
that it does **not** survive. That prediction is recorded here so the outcome
cannot be reinterpreted afterwards.

No arm will be added, dropped or retuned after seeing results.
