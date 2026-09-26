# F4 — adaptive posterior integration and fresh continuous confirmation

2026-09-09. Frozen before new F4 solver observations. F3's raw field/clock
checkpoints remain frozen; the rejected global discrepancy correction stays off.
Manuscript and production inference are unchanged.

## Research basis and method

[SciPy's cubature documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html)
describes global error control through adaptive region subdivision and explicitly
warns that convergence is not guaranteed. Its
[integration tutorial](https://docs.scipy.org/doc/scipy/tutorial/integrate.html)
notes that a narrow important region can be missed. We therefore retain the
complete F3 prior grid as the starting survey, refine on direct-solver
likelihoods, preserve full prior bounds, and check successive meshes. This is a
custom positive dyadic trapezoidal rule, not SciPy's Gauss–Kronrod/Genz–Malik
implementation, and its embedded differences are estimates, not rigorous bounds.

Start with 32x32 cells and the existing 65x65 solver nodes. Each cell compares
its corner trapezoid with the four-child (3x3-node) trapezoid. Normalize each
observation's likelihood by its largest cached log likelihood for numerical
stability. Sum absolute cell-rule differences per observation and divide by
its estimated evidence. Refine up to 64 cells with the largest normalized
local error each iteration. Reuse all node profiles across observations/times.
Every new node is a direct PDE-solver evaluation; no neural surrogate or
interpolated forward field supplies the reference.

Use successively tighter estimated evidence-error tolerances 0.02 and 0.005.
If their empirical convergence checks fail, continue to 0.00125 as a predefined
numerical fallback. Cap at 8,000 new integration nodes and lattice spacing
1/4096 of each prior axis; hitting a cap is explicitly not convergence. Save
node caches, cell identities and stage weights atomically so work is resumable.
The reference refines on solver likelihoods only. Independently check candidate
and production posterior changes on the successive same-bounds meshes.

## Fresh observations and checks

Generate 256 independent continuous-prior parameter draws, seed 980000000;
noise seed 980100000, shared across methods. Keep F3's prior, NFW/Yukawa physics,
20 radii, 0.0434 dex noise and fixed times 100/400/900. Keep only solver survivors
at each time, recording all statuses/failures. No correction or model selection
uses these observations. Numerical mesh adaptation to an integrand is part of
the integration algorithm, not neural-model fitting.

Require per time, for solver and raw candidate, p95 successive posterior mean
shift <=0.1 final posterior SD, median relative 90%-interval width change <=10%,
and p95 changes in posterior CDF at the two true parameters and the solver
log-likelihood quantity <=0.02. Use the last two tolerance levels reached;
report all levels and explicit cap/nonconvergence status. These CDF checks
address quadrature discretization of coverage/rank diagnostics as well as moments.

Then report continuous-truth CDF uniformity (Holm within nine tests per method),
68/90/95% central-CDF coverage with Wilson intervals, ordinary 90% quantile
coverage, paired posterior TV on a common final mesh, and surviving prior mass.
A direct-solver positive control must pass uniformity; a data-ignoring prior
control must fail the likelihood-quantity test. No MCMC samples are used.
Candidate screening retains F3's p>=0.05, parameter 90% central-CDF coverage
0.85–0.95, median TV<=0.10 per time, plus the integration checks above.
Small survivor counts limit power; passing is evidence under this observational
model, not a guarantee of calibration or spatial/physical solver convergence.

Refinement is a numerical procedure and cannot make a failed statistical test
pass by changing model weights, uncertainty, prior bounds, test noise, or
post-hoc exclusions. Preserve all negative findings. Do not replace production
on this experiment alone.
