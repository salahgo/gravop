# F10: clock-feature and gate development ablation

One seed; previously used validation splits. These are development results.

| System | Full | Features only | Gate only | Direct | Interaction |
|---|---:|---:|---:|---:|---:|
| sidm | 0.00330416 | 0.00349685 | 0.00923954 | 0.00922705 | 0.9436 |
| rd | 0.07027 | 0.0701065 | 0.100755 | 0.0966364 | 0.9614 |
| burgers | 0.00141652 | 0.00127751 | 0.00096224 | 0.000741606 | 0.8546 |

Lower nRMSE is better. Interaction below one is favorable on the multiplicative error scale.
All arms use equal tensor shapes and training budgets; feature-zeroed weights are inactive.
No model replaces the frozen F9 candidate. Replication and fresh confirmation are required
before interpreting an apparent improvement as generalization.

[Protocol](FORECAST_FACTORIAL_PROTOCOL.md). Raw records: `results/forecast/factorial_v1`.

## Interpretation and next experiment

Clock features provide the main SIDM/RD benefit in this single-seed pilot.
For SIDM, full conditioning is 5.5% better than features with a physical-time
gate; removing the features degrades both gate choices substantially. For RD,
features-only is 0.23% better than full, a difference too small to interpret
without seed replication. The clock gate alone does not recover the feature
benefit in either system. Late-time SIDM error is slightly lower for
features-only, so full is not uniformly better across the reported metrics.

Burgers favors the direct arm within this same hybrid architecture: its
validation nRMSE is 47.6% lower than full. This direct arm is not the FNO control
from F2, and these validation scores cannot be compared as fresh-test scores
against the F2 table. Interaction below one does not imply that adding either
component improves absolute error; Burgers illustrates that distinction.

The next priority is a separately specified multi-seed replication of the
same four arms. A clock-perturbation training experiment is a later option if
replication supports useful but fragile alignment. These are recommendations;
no additional seeds or candidate replacement followed this 12-fit pilot.

## Verification and provenance

All 12 fits completed. An independent audit checked the original archived
sources and input identities, equal tensor parameter counts within each system,
selection histories, checkpoint identities and the reported interaction. It
also replayed validation predictions from every checkpoint on cuda:0 and
reproduced the stored per-trajectory and per-time scores. See
`results/forecast/factorial_v1/audit_replayed.json` and
`logs/f10_checkpoint_audit.log`.

The repository-wide test run found two administrative failures: the manuscript
still counted 155 rather than 158 tests, and the new tests lacked required
verification decorators. It passed the other 156 tests. The test labels were
added only after all 12 fits completed. The original nine source files are
preserved in `source_snapshot.zip`, and the original manifest hashes are
unchanged. The auditor checks that the test-file AST differs only by those
classification decorators. The corrected gates, new model tests and release
workflow tests pass together: 11 passed (`logs/f10_corrected_gates.log`).

For an exact training replay, use a separate checkout, preserve its shipped
F10 output directory under another name, restore the recorded source snapshot,
and supply the original data and clock prerequisites before running the suite.
Do not restore the old tests into the active calibration workspace. For a
read-only checkpoint audit, run `scripts/audit_forecast_factorial.py --device
cuda:0` under the recorded research CPU limit. These experimental checkpoints
require `FactorialOperator(config['arm'], **config['model'])`; the generic
`ForecastPredictor` does not encode their ablation arm.
