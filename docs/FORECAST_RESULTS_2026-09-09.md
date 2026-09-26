# Forecast model results — 9 September 2026

> Update: [F2 fresh confirmation and full SIDM validation](FORECAST_VALIDATION_RESULTS_2026-09-09.md) supersede the broad F1 recommendation. The original results below are exploratory.

**Implemented and trained a physical-time forecasting model that improves four
of the five benchmark systems.** The strongest result is SIDM: mean held-out
nRMSE falls from 0.007785 to 0.003123, a **59.9% reduction**. The baseline is a
CNO with the same converged clock, chosen on validation data. Manuscript work
remains deferred.

## What changed

`statecond.forecast.ClockConditionedOperator` uses a frozen, well-trained clock
to estimate time relative to the terminal event. Its field network also receives
absolute-time and near-event features, combines Fourier mixing with local
convolutions, and predicts a residual from the initial field. A gate enforces
the initial condition exactly. Field training uses the model's predicted clock,
not an oracle event time supplied as a conditioning feature.

The deployment interface takes **initial fields, physical parameters, and
physical query time**. It never takes measured target phase, a branch flag, or
the true terminal time. A standalone checkpoint includes both networks and all
normalizations. `ForecastPredictor` runs without loading the training dataset.

## Held-out comparison

All values below are **means of three separately trained seeds**, not an
ensemble's prediction error. The same trajectory split is used throughout.
For each system, the reference is selected by mean validation nRMSE from
converged-clock FNO, CNO, Transformer, and direct-time FNO. The test set does not
select the model architecture, checkpoint epoch, or reference.

| System | Selected reference | Reference nRMSE | New nRMSE | Error reduction | Reference/new ratio, paired 95% CI |
|---|---|---:|---:|---:|---|
| SIDM | CNO + converged clock | 0.007785 | 0.003123 | **59.9%** | 2.49 [2.19, 2.86] |
| Reaction–diffusion | CNO + converged clock | 0.071986 | 0.047680 | **33.8%** | 1.51 [1.37, 1.70] |
| Keller–Segel | CNO + converged clock | 0.003544 | 0.002176 | **38.6%** | 1.63 [1.37, 1.96] |
| Thin-film rupture | FNO + converged clock | 0.062962 | 0.057451 | **8.8%** | 1.10 [1.02, 1.18] |
| Burgers | Direct-time FNO | 0.001743 | 0.001765 | **−1.3%** | 0.99 [0.89, 1.10] |

Source: `results/forecast/summary.json`; regenerate with
`python scripts/sc_forecast_suite.py report`. Individual evaluations preserve
per-trajectory errors, per-time errors, split indices and reference checkpoint
hashes. There are 61 test trajectories for SIDM and 60 for each other system.
Bootstrap intervals resample paired trajectories, averaging seeds within each
draw. These are pointwise intervals, without a five-comparison multiplicity
correction, and do not encompass all retraining uncertainty.

Burgers is a tie, with slightly worse point estimates; it does not support
replacing its existing direct-time model. Film's gain is modest. Late-time error
improves by roughly 48%, 25%, 30%, and 2% in SIDM, reaction–diffusion,
Keller–Segel, and film respectively, but worsens by approximately 14% in Burgers.
The overall averages therefore must not be presented as uniform late-time gains.

## What the controlled variants show

SIDM seed-0 validation, same 200 epochs and batch size:

| Variant | Validation nRMSE |
|---|---:|
| Aligned clock + spectral/local field model | **0.003304** |
| Same model with local convolutions disabled | 0.003616 |
| Direct-time variant without clock features | 0.009018 |

The clock-feature ablation changes the time-dependent residual gate as well as
conditioning features, so it supports the combined alignment design, not an
isolated claim about one feature. These are one-seed validation experiments;
their test sets were not evaluated. The selected full recipe was then frozen
for all five systems and seeds 0, 1, 2.

## Cost, scope, and limits

- About 670,000 trainable field parameters; approximately 835,000 total including
  the frozen clock for the single-field systems, and 868,291 total for SIDM.
- Each main field fit took 56–62 seconds on an RTX 4090. The 15 main fits total
  approximately 865 GPU-seconds; independent jobs ran on two devices. The
  previously trained clocks and the additional pilot ablations are excluded.
- The new recipe uses batch 512 versus 1024 in the original field-training
  script: approximately twice as many optimizer updates per epoch. These
  results establish a better model/training recipe, not an architecture-only
  gain under identical compute. The local/clock ablations do use matched budgets.
- A warmed SIDM GPU forward at batch 61 measured 0.836 ms per batch, about
  13.7 microseconds per profile amortized. This excludes data transfer and input
  preparation and is not a single-request latency. See `sidm_latency.json`.
- SIDM here is the **603-trajectory terminal subset**, including collapse and
  Knudsen termination. These numbers do not replace the original full-D1 density
  error percentage or establish an improvement in the production MCMC pipeline.
- Spatial resolution is fixed by the clock encoder. Nonuniform-grid inference,
  new initial-condition families, and out-of-domain parameters are not validated.
- A predicted horizon flag is exposed, but the clock can be wrong; it is not a
  physical-validity certificate. Film's remaining late-time sensitivity is a
  clear target for future work.
- The test split has a history of prior project evaluations. A fresh confirmation
  set after freezing this recipe is the next stronger generalization check.

## Reliability work included

- Corrected the shared benchmark scorer to enter evaluation mode, and tested
  dropout determinism and unchanged BatchNorm statistics.
- Tested exact initial fields, finite backward passes with one/two channels,
  frozen clock gradients, standalone checkpoint inference, and invalid inputs.
- Run directories refuse accidental overwrites. Checkpoints carry source, data
  and clock identities, configuration, split indices, normalization and epoch.
- Older CNO/Transformer/direct-FNO checkpoints without loss metadata are checked
  against stored per-trajectory oracle/direct errors before inclusion. Identity
  checking reproduces original per-time batching; convolution numerical results
  can differ slightly across devices and batch shapes. CPU verification permits
  1% relative tolerance in these small errors; final tables were evaluated on GPU.

The only intentionally deferred suite issue is the Markdown manuscript's
hard-coded test count, now outdated because the new model adds tests. The
manuscript itself has not been edited.

Final validation: **112 passed, 1 explicitly deselected in 150.35 seconds**.
The deselected test is
`test_p2_application.py::test_manuscript_numbers_match_the_artefacts`; a full
run confirmed its failure is the outdated gate count. The four focused model
tests also passed after tightening the public predictor's exact initial-value
handling. `git diff --check` passes.

## Use and reproduce

See [the frozen experiment contract](FORECAST_EXPERIMENT.md) for prerequisites,
commands and the inference interface. The implementation is in
`src/statecond/forecast.py`; training/evaluation are in `scripts/sc_forecast.py`;
the five-system runner and report generator are in `scripts/sc_forecast_suite.py`.

Next, prioritize a fresh confirmation dataset and a controlled transfer to the
full SIDM application, including nonterminal trajectories, observation-space
density errors, calibrated discrepancy, and solver-backed inference checks.
Keep the current production inference ensemble until those checks pass.
