# F3 — inference fixes and population calibration protocol

2026-09-09. Frozen before generating or scoring F3 solver data. Manuscript and
production inference remain unchanged. F2 field/clock checkpoints remain frozen.

## Research and resulting fixes

- [Talts et al.](https://arxiv.org/abs/1804.06788) and the
  [Stan SBC guide](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html)
  motivate prior-predictive, solver-generated observations and independent
  posterior diagnostics. Self-generated emulator observations cannot establish
  agreement with the solver.
- [Modrak et al.](https://arxiv.org/abs/2211.02383) show that parameter-only
  ranks can miss methods that ignore the data. Add the solver log likelihood
  as a data-dependent test quantity and a deliberately data-ignoring control.
- [Sailynoja et al.](https://arxiv.org/abs/2103.10522) motivate careful uniformity
  checks. Use randomized posterior CDF values, resolving discrete ties exactly,
  and Holm-adjusted KS tests rather than visual histogram judgments. Exact
  finite-grid posterior weights avoid posterior-draw autocorrelation.
- [Hermans et al.](https://arxiv.org/abs/2110.06581) motivate coverage and
  informativeness checks together. Report posterior distance and widths;
  broadening intervals alone does not establish an improved approximation.
- [Covariance shrinkage documentation](https://scikit-learn.org/stable/modules/covariance.html)
  motivates regularizing correlated residual covariance. Independently select
  a bias/covariance correction by held-out Gaussian log score. A Gaussian,
  parameter-independent correction is an approximation to test, not a guarantee.

## Frozen design

Use the existing NFW Yukawa application prior: log10 sigma0 in [-1.5,1],
log10 w in [-1,1.5], independent uniform. Observation radii and noise remain
20 radii in [0.02,20] r_s and 0.0434 dex. Use fixed physical times 100,400,900;
never choose a query time from the truth's terminal time.

Generate direct-solver profiles on a 65x65 grid; its 33x33 nested grid has
identical bounds. Trapezoidal integration weights define an explicit discrete
prior on each grid. At each time, condition on the solver reaching that time:
this is the population of halos with a profile still defined by the model.
Record all stops and numerical failures; numerical failures block acceptance.
The learned-clock masks approximate this survival support and remain part of
the deployable candidate. Also report rejected true support and false support.
This experiment is not inference beyond termination or an uncensored population.

Generate 256 separate continuous-prior calibration trajectories (seed 930000000),
using rows 0:192 to fit corrections and 192:256 to select them. Per time and model,
compare: no correction; mean bias only; bias plus residual covariance shrunk
10%,50%,100% toward its diagonal. Add known observational variance in every
likelihood. Selection uses residuals plus independently generated observation
noise, with seed 930100000. Do not refit after selection. Never use final
observations or the old mock to select a covariance or inflate uncertainty.

Generate 128 additional continuous-prior confirmation trajectories
(seed 940000000), all disjoint from calibration. They test grid adequacy and
continuous-truth coverage; they do not select the correction. Observation noise
uses seed 940100000. For finite-grid SBC, generate 512 independent truth/noise
pairs per time (seed 950000000+time index), sampled from the exact surviving
65x65 discrete prior. Evaluate solver, candidate raw/corrected, production
raw/corrected and the prior-only negative control on identical observations.
Parameter coordinates and solver log likelihood are the three test quantities.
Report central randomized-PIT coverage at 68/90/95%, Wilson intervals, KS
statistics/p-values, per-observation TV to the solver, and posterior uncertainty.
Holm adjustment is within each method's nine tests (three quantities x three
times); comparisons between methods remain descriptive.

Check 12 prespecified random grid trajectories with eps_t reduced from 3e-4 to
1e-4 and snapshot spacings halved. Standard runs retain the production solver
and physics. A solver discrepancy over 0.00434 dex (10% of observation noise)
at any valid checked radius/time blocks an accuracy claim. Report discrepancies
in survival support. Check 33/65 grid adequacy on continuous confirmation:
95th-percentile mean shift <=0.1 fine-grid posterior SD and median 90%-interval
width change <=10% per time/parameter. Failures limit interpretation to the
finite-grid experiment and require later refinement, without silently retuning.

## Decision rules

Numerical/input/provenance checks and the solver positive control must pass;
the likelihood test must detect the data-ignoring negative control. Candidate
raw and corrected are both reported. A deployable arm may advance only if all
nine Holm-adjusted tests have p>=0.05, each parameter/time's randomized 90%
coverage is between 0.85 and 0.95, and median TV to the solver is <=0.10 at each
time. The corrected candidate must also have no median-TV regression >0.02
against the raw candidate. These are screening tolerances, not proof of exact
calibration. Numerical-reference/grid checks must pass before extending claims
to the continuous prior. Do not replace production on this experiment alone.
