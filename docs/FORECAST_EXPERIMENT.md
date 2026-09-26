# Physical-time forecast improvement — experiment contract

> Update: [F2 fresh confirmation and full SIDM validation](FORECAST_VALIDATION_RESULTS_2026-09-09.md) supersede the broad F1 recommendation. The original results below are exploratory.

2026-09-09. Manuscript work is deferred. This experiment improves deployable
field predictions using only initial fields, physical parameters, and query time.

## Hypothesis and model

A well-trained scalar clock provides useful alignment, but a field operator
trained only on oracle normalized time is not optimized for the deployed query.
Train on physical-time queries through a frozen, converged clock. Supply both
absolute time and features resolving rapid variation near the estimated terminal
event. Combine global Fourier mixing with local convolutions and predict a
residual from the initial field, multiplied by a gate that vanishes exactly at
time zero. The model does not receive observed target phase or branch labels.

The clock is reused from the existing converged FNO clock checkpoint for the
corresponding system and seed, with normalization verified before use. Its prior
training cost is additional to the field fit and must remain explicit. This is
a new forecasting model, not a claim of a new generic neural-operator family.

## Selection before test evaluation

- Trajectory split: existing seed 1234, 80/10/10. Normalization uses training only.
- SIDM means the 603 terminal trajectories in `data/sc/sidm.h5`, not the full
  1300-trajectory application dataset. No production MCMC integration is claimed.
- Initial pilots: SIDM and reaction–diffusion, seed 0, 200 field epochs.
- SIDM ablations: identical budget with clock features removed, and with local
  convolutions removed. Only validation metrics are inspected during selection.
- Freeze the aligned local/spectral recipe for five systems and seeds 0, 1, 2.
  Width 64, four blocks, 32 Fourier modes, batch 512, AdamW lr 1e-3, cosine
  schedule, gradient clipping at 1, field MSE, 200 epochs. Select the checkpoint
  with minimum validation nRMSE at the fixed 20-epoch evaluation intervals.
- Test evaluation is a separate CLI stage after freezing this choice. The
  reference is the best available saved time-based arm: converged-clock FNO,
  CNO or Transformer, or direct physical-time FNO, selected by mean validation
  nRMSE across seeds for each system. Each aligned reference uses the identical
  frozen converged clock. DeepONet checkpoints are unavailable and excluded.
- Report absolute mean nRMSE, late-time error, seed variation and a paired
  trajectory bootstrap of reference/model error. The bootstrap averages seeds
  and estimates trajectory uncertainty, not the full uncertainty of retraining.
- Record failures and systems without improvement. Do not tune on test results.

The original direct-time baseline floors its zero-time input using the true
trajectory event time. The evaluator reproduces that historical convention only
for the reference. The new model requires no such event time at inference and
returns the initial field exactly. All methods' query times are constructed on
the established evaluation grid for comparability.

## Reproduction

Requires generated `data/sc/*.h5` and the existing clock/oracle/direct checkpoints
under `results/statecond/ckpt`. A missing prerequisite fails rather than falling
back to the short-budget clock. To regenerate clocks, the existing command is
`python scripts/sc_timehead.py --all --budgets 20000 --save-heads` after the
original operators have been trained and saved.

```bash
python scripts/sc_forecast_suite.py train --devices cuda:0 cuda:1 --resume
python scripts/sc_forecast_suite.py evaluate --devices cuda:0 cuda:1
python scripts/sc_forecast_suite.py report
```

Outputs live under `results/forecast`; each run keeps a configuration, dataset
and clock hashes, source hashes, split indices, validation history, a standalone
checkpoint, and separate held-out evaluation. Existing scientific artifacts are
not overwritten. Weight files are gitignored; metrics and recipes are retained.

```python
from statecond.forecast import ForecastPredictor
predictor = ForecastPredictor('results/forecast/sidm_aligned_s0/model.pt')
# Initial fields: (B, 2, 128), log10 density and log10 velocity.
# Parameters: (B, 3), log10 sigma0, log10 w, model_id in that order.
result = predictor.predict(initial_fields, parameters, physical_time)
fields = result['fields']
```

The horizon flag is based on the learned clock and is not a certificate of
physical validity. Spatial resolution is fixed by the clock encoder. Accuracy
outside the training parameter domain has not yet been established.
