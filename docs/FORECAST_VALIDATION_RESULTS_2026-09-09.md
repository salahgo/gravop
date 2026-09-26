# F2 — fresh confirmation and full SIDM validation

> Follow-up: [F3 population calibration and inference fixes](FORECAST_CALIBRATION_RESULTS_2026-09-09.md) retain the raw candidate, reject a selected global discrepancy correction, and identify posterior-grid convergence as the next limitation.

2026-09-09. Research and validation of the F1 recommendation, with the manuscript deferred. Protocol: [FORECAST_VALIDATION_PROTOCOL.md](FORECAST_VALIDATION_PROTOCOL.md).

## Decision

Advance the new physical-time forecaster as the SIDM candidate for broader inference validation. It passed all seven predeclared application screens. Keep the current production inference pipeline until population-level posterior calibration is tested. Retain the direct-time FNO for Burgers; the new architecture is not a universal improvement. These results supersede the broader interpretation of the exploratory F1 gains.

## Research basis and controls

Explicit experimental records follow the reproducibility principles in [Pineau et al. (2021)](https://www.jmlr.org/papers/v22/20-303.html). The use of independent trajectories and seed sensitivity follows the evidence on benchmark variance in [Bouthillier et al. (2021)](https://arxiv.org/abs/2103.03098). The distinction between forward accuracy and posterior calibration follows [Talts et al. (2018)](https://arxiv.org/abs/1804.06788). The specific budgets and screening tolerances are project choices, recorded before scoring.

Fifteen reference models were retrained: the previously validation-selected CNO for SIDM/RD/KS and FNO for film/Burgers, three seeds each. They match F1 at 200 epochs, batch 512, optimizer updates and examples, with the same AdamW/cosine recipe and validation-selection schedule. This is not a FLOP-matched comparison or a new exhaustive search of all backbones. Aligned references train on oracle tau and deploy the same frozen clocks used by F1. The direct Burgers reference uses log1p of physical time divided by a training-only time scale, removing the historical oracle-dependent zero-time floor. Thus its improvement over the old reference includes a coordinate correction.

Fresh confirmation used 1,600 new solver draws: 300 per bespoke PDE and 400 SIDM. There were no solver failures; one RD and seven Burgers draws were nonsingular and excluded from the terminal benchmark. SIDM retained all 400 for application evaluation: 79 density-stop, 109 Knudsen-stop and 212 horizon-stop trajectories; 188 enter the terminal benchmark. New seed blocks and exact input-identity checks found zero overlap. F1 weights, controls, normalization and checkpoint selection were frozen before fresh scoring.

## Fresh benchmark results

Errors are mean per-snapshot root-mean-square errors in training-standardized fields, averaged over time, trajectories and three seeds. They are not raw density percentages and not ensemble errors. Ratios above one favor F1. The intervals below resample both seeds and paired trajectories (5,000 draws); only three seeds are available. They are pointwise sensitivity intervals, not multiplicity-adjusted claims.

| System | Fresh trajectories | Matched reference | F1 | Error reduction | Reference/F1, seed + trajectory 95% interval |
|---|---:|---:|---:|---:|---|
| SIDM | 188 | 0.006238 | 0.004110 | +34.1% | 1.518 [1.423, 1.635] |
| RD | 299 | 0.067192 | 0.054618 | +18.7% | 1.230 [1.111, 1.436] |
| KS | 300 | 0.002780 | 0.002892 | -4.0% | 0.961 [0.911, 1.024] |
| film | 300 | 0.055456 | 0.053452 | +3.6% | 1.038 [0.962, 1.118] |
| burgers | 293 | 0.000933 | 0.001619 | -73.5% | 0.576 [0.513, 0.639] |

SIDM and RD improve under the stricter comparison. KS is 4.0% worse on the point estimate: its trajectory-only interval favors the control, but the seed-sensitive interval crosses one. Film’s 3.6% improvement is inconclusive. Burgers is 73.5% worse and decisively favors the direct FNO. No candidate hyperparameters were changed after these results.

| System | Control field training, mean seconds | F1 field training, mean seconds | Fresh reference/F1 from control wall-time prefix |
|---|---:|---:|---:|
| sidm | 115.5 | 61.3 | 2.842 |
| rd | 109.3 | 56.0 | 2.068 |
| ks | 108.3 | 57.2 | 1.446 |
| film | 37.0 | 57.0 | 1.038 |
| burgers | 37.1 | 57.0 | 0.576 |

These elapsed times include training and scheduled validation on the available GPUs; they are descriptive, not controlled hardware timing trials. Frozen-clock pretraining is an additional shared cost. The prefix chooses the best validation checkpoint available within the previously recorded F1 wall time; it is not an independently optimized short-budget run. F1 has about 670k trainable field parameters versus 577k–587k for the controls. It runs faster than these CNO controls and slower than these FNO controls.

## Full SIDM transfer

Three new clocks and field models were trained on the full 1,300-run D1 population. The original production split is preserved: 1,040 training, 130 validation, 130 test trajectories. Reusing the terminal-subset clock would leak 49 full-D1 validation and 46 test rows; the new clocks train only on the full-D1 training partition and have zero overlap. Clock training used the fixed 20,000-epoch, batch-256 recipe on CPU; field training used unchanged F1 settings for 200 epochs on cuda:1.

The candidate averages three models; production averages its existing five models. This is an application comparison of complete systems, not a matched-budget architecture ablation: the clock, field model, training recipe and ensemble size differ. The production adapter agrees with the existing inference implementation within 1.06e-5 dex on the audited inputs.

| Population | n | Production mean density error | Candidate mean density error | Production observation RMSE (dex) | Candidate observation RMSE (dex) |
|---|---:|---:|---:|---:|---:|
| Original D1 test, all | 130 | 6.975% | 2.488% | 0.04546 | 0.02685 |
| Original D1 test, terminal | 59 | 13.976% | 4.893% | 0.06685 | 0.03969 |
| Original D1 test, horizon_stop | 71 | 1.156% | 0.488% | 0.00836 | 0.00341 |
| Fresh SIDM, all | 400 | 9.118% | 1.406% | 0.04935 | 0.01936 |
| Fresh SIDM, terminal | 188 | 17.798% | 2.351% | 0.07052 | 0.02751 |
| Fresh SIDM, horizon_stop | 212 | 1.421% | 0.567% | 0.01362 | 0.00598 |

Fresh mean density error decreases by 84.6%; the paired trajectory reference/candidate ratio is 6.49 [4.73, 8.46]. This interval conditions on the two fixed ensembles. Heavy tails influence the mean: median per-trajectory density error improves from 1.519% to 0.625%.

Raw density error is mean absolute relative error across 128 radial cells and 48 stored times. Observation RMSE is measured in log10 density on the existing 20 observation radii, 0.02–20 r_s. Predictions are scored for every query, including queries past a predicted horizon; none are silently dropped from field metrics. The fresh predicted-horizon rejection fraction falls from 3.16% to 1.28%. These are sampled-query fractions, not trajectory failure rates. Horizon-stop times are censored endpoints, not measured collapse times.

## N-body and solver-backed inference screens

| Screen | Production | Candidate | Result |
|---|---:|---:|---|
| N-body pooled log-density RMSE, dex | 0.02316 | 0.02021 | Pass |
| Posterior total variation, grid_logpost.npz | 0.23898 | 0.04642 | Pass |
| Posterior total variation, grid_zoom.npz | 0.24884 | 0.09454 | Pass |

Together with four fresh density/observation stratum checks, all seven prespecified screens pass. N-body uses the five existing snapshots and fixed time rescaling 0.945; this is not an independent calibration set, and the candidate is slightly worse at the earliest snapshot. The pooled improvement is about 12.8%.

Posterior comparisons use the same fixed mock and observational-noise-only Gaussian likelihood as the direct-solver grids for both models. They do not compare against the production likelihood augmented with its calibrated model discrepancy. They measure discrete posterior agreement on each stored grid’s own support; the 31-by-31 and 91-by-91 grids differ in extent and are not a grid-convergence test. Direct-solver recomputation at each grid’s maximum exactly reproduced the stored log likelihood. Neither model rejected any nonzero solver posterior mass on these grids.

The next recommended experiment is a preregistered simulation-based calibration study across the prior and multiple observation times, with independently selected discrepancy calibration and solver-backed rank/coverage checks. Preserve a new untouched confirmation set. The two fixed grids do not establish population coverage, MCMC convergence, robustness outside the training distribution, or improved accuracy of the underlying physical solver. No production replacement was made.

## Reproduction and verification

```bash
.venv/bin/python scripts/forecast_validate.py generate --workers 16
.venv/bin/python scripts/forecast_validation_suite.py controls --devices cuda:0 cuda:1
# The original full-data clocks were trained on CPU:
for seed in 0 1 2; do
  .venv/bin/python scripts/forecast_validate.py clock --seed "$seed" --device cpu
done
.venv/bin/python scripts/forecast_validation_suite.py application --devices cuda:1
.venv/bin/python scripts/forecast_validate.py confirm --device cuda:0
.venv/bin/python scripts/forecast_application.py --device cuda:0
.venv/bin/python scripts/forecast_validation_audit.py --device cuda:0
.venv/bin/python -m pytest tests/ -q --deselect=tests/test_p2_application.py::test_manuscript_numbers_match_the_artefacts
```

Generation and individual training stages refuse to overwrite existing data/checkpoints; use a clean checkout with the required original data and pretrained dependencies to repeat training. The suites skip completed runs. Evaluation and audit can be rerun against the existing artifacts. F1 training prerequisites are documented in [FORECAST_EXPERIMENT.md](FORECAST_EXPERIMENT.md).

Validation: **115 tests passed, one manuscript-count test deselected**; the count is stale after adding verification tests and manuscript edits remain deferred. The independent audit passed split identity, input overlap, checkpoint/data hashes, completed training budgets, production prediction agreement, and direct-solver likelihood spot checks. `git diff --check` passed. Checkpoints and generated HDF5 files remain local and gitignored.

Artifacts: `results/forecast_validation/fresh_manifest.json`, `confirmation/*.json`, `controls/*/{config,training}.json`, `application/validation.json`, `application/{implementation_audit,solver_grid_audit}.json`, and `execution_provenance.json`. Per-trajectory errors, bootstrap intervals, checkpoint hashes and screening decisions are stored in JSON; logs are in `logs/f2_*`. Manuscript and production checkpoints are unchanged.
