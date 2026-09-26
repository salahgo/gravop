# Bounded GPU acceleration validation

Opened on 11 September 2026. F8 numerical sources and results stay frozen.
Its last CPU diagnostic continues; the waiting F9 launcher is paused until the
acceleration decision.

Implement only the existing 300-shell, implicit conduction, Yukawa-tu index-5
reference configuration, with FP64 arithmetic and no fast math/FMA. Preserve
adaptive step criteria, hydrostatic damping, physical stop conditions and
observation interpolation between the original snapshot times. The prototype
returns inference profiles and final states, not a replacement for the solver's
complete energy/snapshot API. A serial-Thomas CUDA prototype provides an
arithmetic comparison; parallel cyclic reduction may replace its linear algebra
only after CPU agreement checks. This does not change the PDE discretization.

Check initial/perturbed hydrostatic updates and capped short trajectories first.
Then validate complete trajectories at all three observation times on the six
F8 truths, all four prior corners, and 22 independent prior draws using seed
992000000. Require identical terminal status and survival masks, maximum common
profile difference <=1e-5 dex, and relative terminal-time difference <=1e-6.
Retain failed checks. Repeat the six F8 truths at stricter solver level 2. Check
both GPUs. This is a development validation bank, not fresh population SBC.

Measure warm throughput on 128 independent prior draws from seed 992000001,
against 24 warmed CPU workers. Keep compilation/spawn costs separate and report
host/device transfer overhead. Suspend the owned F8 process group only while
measuring the CPU baseline to avoid contention; always resume it, including on
errors. Compare one GPU and both GPUs splitting the same work. Also report the
smaller correctness batch and imbalance due to differing trajectory lengths.
A candidate needs at least 2x warm throughput versus the 24-worker CPU baseline,
with all numerical gates passing, to justify integration work. Otherwise retain
the negative performance result and resume the original validated CPU method.
Do not claim a calibration speedup from a neural-only benchmark.

Before adopting a passing implementation, validate complete posterior summaries
against existing CPU reference cases, retaining the original mean/width/CDF and
temporal gates. Freeze the accelerated execution and its provenance separately;
never relabel old CPU artifacts as GPU runs or relax calibration gates. Reconcile
the release plan and tests before the F9 launcher resumes.

References: [NVIDIA CUDA programming](https://docs.nvidia.com/cuda/cuda-programming-guide/),
[NVIDIA GPU execution recommendations](https://docs.nvidia.com/cuda/ada-tuning-guide/),
and [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

---

*Publication note.* Sentences recording how this study was authorised and scheduled have been reworded for publication; nothing the protocol commits to has changed. The experiment records name the frozen text by its SHA-256, `37f5acf6759b3c4674f1a5c175f0bf202c016fd2a877ddf9ad9825fff13d90a9`.
