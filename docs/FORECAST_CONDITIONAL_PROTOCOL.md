# F6 — conditional posterior integration pilot

2026-09-10. Frozen before F6 solver evaluations. Preserve F4/F5 failures and all
model checkpoints. This is a numerical pilot, not a population calibration study.

## Scope and research basis

Use F5's difficult interior-time case: F4 observation 353, time 400. It failed
F5's integration and CDF checks. Choosing an observation before the solver's
900 time cap permits direct bracketing of its survival transition without
changing the reference trajectory horizon. Keep the original full rectangular
log-sigma0/log-w prior, observations, noise and level-1 solver settings
(eps_t=1e-4, snapshot spacings 0.01/0.025, t_end=900). Reuse only verified F5
level-1 profiles and its recomputed truth likelihood. No discrepancy fitting.

[QUADPACK's breakpoint interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
supports explicitly locating discontinuities. The implementation uses custom
positive 3/5-point Gauss-Legendre rules inside each conditional integral, and
nested Simpson rules outside. [Outmezguine et al.](https://arxiv.org/abs/2204.06568)
motivate investigating clock-related narrow likelihood regions through their
approximate temporal universality result. We locate peaks directly in log sigma0
at each log w; we do not assume an exact physical scaling or transform the prior.

## Inner integrals

At fixed log w, survey 33 equally spaced log-sigma0 values over the full prior.
Bracket every observed solver and candidate survival transition; root-find to
1e-7 in log sigma0. Optimize each sampled local likelihood maximum separately
for both methods, with a maximum of 12 candidate modes per method (exceeding
that limit is a diagnostic failure). Locate shoulders at a log-likelihood drop
of 12. Split at these peaks/shoulders, support boundaries, the true sigma0,
and all detected crossings of the solver log likelihood at its true value.

Integrate evidence, normalized first/second sigma0 moments, and both sigma0 and
solver-likelihood CDF numerators. Adapt on raw absolute 3/5-rule differences,
normalized by each method's conditional evidence. If a likelihood-CDF indicator
still changes within a panel, charge its full mass as indicator uncertainty.
No tail truncation, interpolated forward reference, or missing-support relabeling
is allowed. The finite survey is not a mathematical guarantee that all possible
hidden modes or survival islands are detected; retain that limitation.

## Outer integrals and gates

Initial log-w panels split the full prior at 0.25 and the true log w. Integrate
conditional values with positive 3/5-node Simpson rules. Classify the w-CDF per
panel, avoiding artificial endpoint atoms at the truth. Propagate inner absolute
errors through the positive outer weights. Require all evidence, moment and CDF
errors, relative to the respective method's joint evidence, <=0.02 then <=0.005;
inner tolerances are one tenth of each outer target. The predefined third target
is 0.00125 if successive posterior summaries fail. Bisect every outer panel
between tolerance stages, retaining the entire prior.

Both solver and frozen candidate must satisfy successive mean changes <=0.1
final SD, relative 90%-interval width changes <=0.1 and changes in all three
truth CDFs <=0.02. Widths use positive-node marginal quantiles and must be
reported as numerical approximations. Record the paired posterior TV but do
not interpret it as calibrated population inference.

Cap at 8,000 new level-1 solver trajectories, at most 512 new trajectories per
conditional column, and 256 inner panels. Existing verified level-1 nodes do not
consume the new-solve budget. A cap, optimizer/root failure, numerical solver
failure or unresolved support invalidates convergence. Save per-column solver
data frequently and all completed rules so an interruption cannot discard a large
completed batch. Reuse is guarded by data, source, solver-setting and model hashes.

Before recommending wider validation, repeat direct-solver likelihood peaks
from five evenly spaced informative outer columns at level 2 (eps_t=3e-5),
requiring unchanged survival and maximum profile change <=0.00434 dex. Analytic
tests cover narrow/multiple conditional modes, survival boundaries, CDF splitting,
positive weights, and propagation of inner error. Keep production and manuscript
unchanged. Only a passing numerical pilot supports a separately frozen broader
study with fresh observations; it does not itself establish calibration.

---

*Publication note.* Sentences recording how this study was authorised and scheduled have been reworded for publication; nothing the protocol commits to has changed. The experiment records name the frozen text by its SHA-256, `c87ed0bee60b7ba246c0d5a8181de956c24a3023153ece740c73d7c73f2359bf`.
