# F8 — final validation and research stopping rule

The model, full log prior [-1.5,1] x [-1,1.5], observational noise 0.0434 dex,
solver level 1 and F7 acceptance thresholds remain fixed. No manuscript or
production claim depends on a capped or incomplete calculation passing.

First validate the six existing F5 observations (worst and median F4 integration
error at each of times 100, 400, 900). These are development diagnostics, not
independent population tests. Case indices run in order 0,4,3,1,5,2, exposing
early, late and intermediate times first. Case 2 is the exactly matching F7
observation and reuses its verified passing result. Before integration, check
all six generating truths at level 2, comparing all common stored times with
level 1; require maximum profile difference <=0.00434 dex and unchanged support.

Generalize F7 to the observation's actual time. New observations perform fresh
support/mode/CDF discovery; never reuse another observation's inner partition.
Exact solver profiles may be reused from the verified F5/F6/F7 banks because
they store all three physical times. New values are durably cached per case.
Start with roots at prior bounds, midpoint and truth log-w. After the first
complete outer assessment, localize the same observed evidence/CDF features
used in F7, once; retain all regions and subsequent error-driven refinement.

Use targets 0.02 then 0.005, inner targets divided by ten, forced bisection of
every outer panel between levels, maximum mean change 0.1 final SD, relative
90%-width change 0.1 and every CDF change 0.02. The predefined 0.00125 fallback
applies only if summary stability fails. Each new diagnostic retains F7's
80,000 submission / 1,024 per-column / 256 inner-panel ceilings. Five posterior
peak level-2 checks and arithmetic audits follow each converged case. Record
failures, partial results and reserved versus completed work explicitly.

Stop this final validation phase at the first required numerical or temporal
failure. Remaining diagnostics and fresh calibration are then recorded as not
run because their prerequisite failed. Do not modify the method, enlarge caps,
select easier replacements or launch another architecture search in this phase.
A negative result closes the research with a scoped inference claim.

Only if every diagnostic passes, freeze the generalized method and generate
256 new independent uniform-prior trajectories (seed 980000000), with independent
Gaussian observations at each surviving time (seed 980000001). Verify no exact
or float32 parameter overlap with D1 or prior calibration/confirmation sets.
Use the same integration and per-case numerical gates, report rank uniformity
for both parameters and solver likelihood, Holm adjustment across nine tests
per method, 68/90/95% coverage with Wilson intervals, posterior TV and support
errors. Any numerical failure blocks a population-calibration claim; do not
discard difficult observations. No uncertainty correction or neural tuning.

This ordering follows [Stan's SBC guidance](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html)
and [Modrak et al.](https://arxiv.org/abs/2211.02383): independently simulated
observations and data-dependent likelihood quantities are required to probe
calibration beyond parameter ranks. Finite test power is reported as a limit.

F7 required approximately 3.5 hours on 24 workers for one case. A successful
full population extension may require substantial computation; cached solver
reuse is measured, not assumed to remove that cost. Failure of a prerequisite
is a scientifically justified stopping point, not a passing calibration result.
