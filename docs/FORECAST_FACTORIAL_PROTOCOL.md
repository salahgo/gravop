# F10 development experiment: separate clock features from the residual gate

11 September 2026. Prospective development protocol, recorded before training.
Additional experiments were opened while continuous-prior calibration
continues with reduced CPU use. F9's models, data, sources and gates stay frozen.

The earlier `aligned=False` ablation removed clock conditioning and changed the
residual gate together. It therefore did not isolate the effect of alignment
features. This experiment separates those interventions in a 2-by-2 design.

| Arm | Clock conditioning features | Residual gate |
|---|---|---|
| full | present | predicted-clock ratio |
| features_only | present | physical time / training time scale |
| gate_only | zeroed | predicted-clock ratio |
| direct | zeroed | physical time / training time scale |

All arms retain absolute log1p time, local and spectral blocks, the same tensor
shapes and parameter counts, and the exact initial-condition constraint. Three
zero feature slots in the nonconditioned arms preserve dimensions but have
inactive input weights; equal tensor size does not imply equal effective
capacity. The frozen clock is evaluated in every arm for consistent overhead.
Both gates use `-expm1(-10 * ratio)`. Full-arm predictions must reproduce the
original model when weights agree; all arms must return the initial field at
time zero. There is no oracle time or measured target state in model inputs.

Fixed pilot: SIDM, reaction-diffusion and Burgers, seed 0, all four arms, exactly
12 fits. These systems include two earlier successes and an earlier failure;
this is a development selection informed by earlier results, not fresh
confirmation. Use existing training and validation trajectories only; no test,
F2 fresh confirmation, F3 or F9 observations are scored. SIDM is the existing
terminal subset, not the full application population.

Each fit uses the original seed-0 converged clock and training-only
normalization, width 64, four blocks, 32 modes, batch 512, AdamW lr 0.001,
weight decay 0.00001, cosine schedule, gradient clipping 1 and 200 epochs.
Every arm resets the same field initialization and sample-order seed. Select
the minimum validation nRMSE at epoch 1 and each 20 epochs. Record mean and
late-time validation errors, per-trajectory errors, selected epoch, training
time, parameter counts, source/data/clock hashes and all failures. Run on one
4090, at nice 10 with two CPU threads inside the research CPU affinity set.
No hyperparameter sweep or automatic additional seeds follows this pilot.

Report every arm and the multiplicative interaction
`E_full * E_direct / (E_features_only * E_gate_only)`, descriptively. Values
below one indicate a favorable joint interaction on this error scale. One
seed and a repeatedly used validation set cannot establish an architecture
ranking or a causal physical mechanism. A changed candidate would require
multiple seeds and separately frozen fresh confirmation, without reusing F9.

Research basis: [FiLM](https://arxiv.org/abs/1709.07871) supplies the existing
conditioning primitive; [U-FNO](https://arxiv.org/abs/2109.03697) demonstrates
prior work combining convolutional and Fourier components. Neither makes the
present architecture a new generic operator family. The paired design and
explicit single-seed limitation follow the benchmark-variance concerns of
[Bouthillier et al.](https://arxiv.org/abs/2103.03098). The hypothesis that the
gate contributes separately is a project-specific inference from the code.

Reproduction: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python
scripts/forecast_factorial.py suite`. Existing run directories are never
overwritten; complete runs are checked before resuming. Outputs are under
`results/forecast/factorial_v1`; weights remain local under the existing ignore
rule. Failure stops the suite and is recorded, without a replacement run.

---

*Publication note.* Sentences recording how this study was authorised and scheduled have been reworded for publication; nothing the protocol commits to has changed. The experiment records name the frozen text by its SHA-256, `c15778d7af9a27fd4455eefd2597ebff964ca87dfe24691f068857664e6312e8`.
