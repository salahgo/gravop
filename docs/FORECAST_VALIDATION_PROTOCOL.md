# F2 validation protocol — frozen before new comparisons

2026-09-09. User direction: research and validate the F1 recommendation first;
manuscript remains deferred and the production inference pipeline stays in place.

## Research basis

[Pineau et al., JMLR 2021](https://www.jmlr.org/papers/v22/20-303.html) motivate
explicit reproducible experimental workflows. [Bouthillier et al., MLSys 2021](https://arxiv.org/abs/2103.03098)
show that data sampling, initialization and hyperparameter choices can affect
benchmark conclusions. Consequently, three initializations on one reused test
partition do not finish validation. [Talts et al.](https://arxiv.org/abs/1804.06788)
describe simulation-based calibration for posterior algorithms; accurate forward
predictions alone do not establish calibrated inference. These references inform
the protocol; the concrete budgets and acceptance rules below are our choices.

## Frozen experiments

1. **Matched optimizer effort:** retrain the previously validation-selected
   time-based reference: CNO for SIDM/RD/KS, FNO for film, direct-time FNO for
   Burgers. Seeds 0–2, 200 epochs, batch 512, AdamW lr 1e-3, cosine decay,
   weight decay 1e-5, gradient clipping 1, validation scoring every 20 epochs.
   Aligned references retain their original oracle-tau training task and use
   the same converged clocks at deployment. Direct time uses log1p(t/training
   time scale), which has no oracle-dependent t=0 floor. Checkpoint selection
   uses deployed validation nRMSE. This matches updates and examples, not FLOPs;
   record actual elapsed training times and the best validation checkpoint
   available within F1's recorded wall-time budget as an additional anytime
   diagnostic. Do not call that prefix an independently tuned time-budget run.
2. **Fresh confirmation:** generate 300 draws for each bespoke PDE and 400 full
   SIDM draws, all using new, disjoint seed blocks. Preserve failures and status
   counts; keep singular trajectories for the benchmark comparison. Freeze F1
   models and matched controls before scoring fresh data; fit no normalization,
   checkpoint selection or hyperparameters on this set. Retain nonterminal SIDM
   draws for the application test. Compute pointwise paired trajectory 95% CIs
   and seed/trajectory resampling as a sensitivity check. Report all systems.
3. **Full SIDM application:** preserve original full-D1 split seed 1234. Train
   a NEW clock on its training partition only: TimeHead, 20,000 epochs, batch
   256, original clock AdamW/cosine recipe, fixed final budget. Do not reuse the
   terminal-subset clock: its training set can overlap full-D1 held-out runs.
   Train the unchanged F1 field architecture for 200 epochs/batch 512 on all
   accepted training statuses 0/1/2, seeds 0–2, validation-selected checkpoints.
   Evaluate the original full-D1 held-out partition and the fresh 400 draws
   against the existing production ensemble; stratify terminal vs horizon-stop
   trajectories and evaluate density on the actual observation radii.
4. **Application decision:** retain production if density or observation-space
   accuracy regresses materially, errors become nonfinite, or solver-backed
   posterior/N-body checks fail. Passing forward-error tests alone does not
   authorize a calibrated-inference claim. Any further posterior checks use
   solver references and fixed likelihood assumptions, not a noise model tuned
   on the final test observations.

   Before application scoring, define "material" as >5% worse mean density
   relative error or observation-space RMSE in either fresh SIDM stratum
   (terminal/horizon-stop). N-body pooled RMSE must not exceed the existing
   ensemble by more than 0.002 dex on the same fixed epochs/calibration.
   Fixed-mock posterior total variation must not exceed production by more than
   0.02 on either stored solver grid. These are screening tolerances, not claims
   of calibrated inference; passing them would justify a subsequent broader
   inference-calibration study rather than immediate production replacement.

Fresh seeds: bespoke systems start at 910,000,000 with offsets of 100,000;
full SIDM starts at 920,000,000. All generation uses the unchanged existing
production solver configurations so the numerical reference definition remains
compatible with the training data. Data and code hashes, split identities,
clock dependencies, configurations, and per-trajectory errors are recorded.

## Interpretation rules

- F1's existing test results have been seen and are exploratory. Fresh-set
  results are the confirmation of the frozen recipe, conditional on the
  existing training data and three trained seeds.
- Retraining only the prior selected backbone does not constitute an exhaustive
  new hyperparameter search across every model family. Report that scope.
- Equal epochs, equal updates, equal wall time, and equal FLOPs are different
  comparisons. Label each measure precisely.
- A tie or negative result changes the recommendation. Do not tune the candidate
  after seeing fresh-set results within this experiment.
- Raw clock error for horizon-terminated SIDM is an approximation to a censored
  terminal time, not a measured collapse time.
