# GPU solver evaluation — 11 September 2026

**Retain the CPU solver.** The GPU prototype passed all CPU-agreement checks but
failed the prespecified throughput requirement. This is a negative performance
result, not a calibration failure and not a claim that no other GPU algorithm
could be faster.

| Implementation | 128 trajectories, seconds |
|---|---:|
| 24 warmed CPU workers | 48.335 |
| One RTX 4090, kernel interval | 126.711 |
| Two RTX 4090s, combined host-to-host interval | 125.253 |

The two-GPU speedup was **0.386x**, below the fixed >=2x adoption gate;
it took 2.59 times the CPU wall time. Compilation/worker startup were
excluded. All 24 CPU workers were warmed, and the owned F8 process group was
suspended and resumed around the CPU measurements to avoid contention. GPU
measurements ran alongside the CPU diagnostic. This is one fixed batch and
hardware configuration, not an exhaustive tuning study. A fair deployment
comparison must include transfer costs; the one-GPU number excludes those costs
and is already slower than the CPU baseline.

![Measured throughput](../results/gpu_solver/throughput.png)

Accuracy checks used six F8 truths, four prior corners and 22 independent prior
parameters at level 1; all six truths were repeated at level 2 on both GPUs.
The additional 128-trajectory throughput batch also passed CPU agreement.
Maximum common observation-profile difference was **1.899e-10
dex**, versus the 1e-5 dex gate. Maximum terminal-time relative difference was
**4.943e-09**, versus 1e-6. Terminal statuses and
survival masks agreed in every comparison. These checks include all three
observation times, snapshot interpolation and early physical termination.

The prototype uses one CUDA block per halo, FP64 without fast math/FMA and
parallel cyclic reduction for the implicit/hydrostatic linear systems. The
physical discretization, step controls and stopping conditions remain unchanged.
An initial hydrostatic transcription sign defect was caught by the one-step
CPU comparison and fixed before the full trial. The serial-Thomas prototype and
failed compilation/accuracy logs are preserved alongside the final passing
accuracy and negative throughput results.

The adapter is experimental and covers only the fixed 300-shell implicit
Yukawa-tu/index-5 inference configuration. It does not implement the complete
CPU solver's energy and snapshot-output API and is not used in calibration or
production. Larger batches or another parallel algorithm could change throughput;
that was not established here. No posterior-level GPU validation is claimed:
the performance prerequisite failed, so that adoption stage was not run.

Frozen protocol: [GPU_SOLVER_PROTOCOL.md](GPU_SOLVER_PROTOCOL.md).
Raw arrays, source identities and independently recomputed gates:
`results/gpu_solver/pcr_trial/`, `results/gpu_solver/audit.json`.
Reproduce the artifact audit with `python scripts/audit_gpu_solver.py`; repeating
the GPU experiment additionally needs CUDA-enabled PyTorch, NVIDIA cuda-bindings
and NVRTC (tested with PyTorch 2.13.0+cu130 and CUDA 13.0).

The fresh CPU confirmation excludes every GPU-development parameter at exact
and float32 precision and freezes that exclusion bank. Earlier F8 sources and
results are preserved. The population calibration continues under its original
numerical gates.
