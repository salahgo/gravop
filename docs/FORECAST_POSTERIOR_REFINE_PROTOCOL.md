# F7 — posterior convergence repair

Frozen before F7 direct-solver submissions. F6 remains a failed, preserved
experiment. This revision addresses the need to fix posterior
convergence; it is not a new population-calibration study.

## Fixed scientific target

Reuse the same F4 case 353 at time 400, observation, likelihood noise, complete
log-sigma0/log-w prior, F5 level-1 solver and frozen raw candidate. Verify all
F6 cached-array hashes and original source/data/checkpoint identities before
reusing a value. Save all new arrays under `data/d1/forecast_posterior_refine`
and results under `results/forecast_posterior_refine`. No F6 cache modification.
Production, model checkpoints and manuscript remain unchanged.

## Algorithm changes

Outer nodes use fixed base panels and integer dyadic descendants, including
saved tree identities. Inherited coordinates are bit-identical. Begin with
the last complete F6 partition. From the completed F6 columns, identify
adjacent columns where either parameter/likelihood conditional CDF changes by
at least 0.2, or log conditional evidence changes by at least 4. A feature is
selected only when an endpoint evidence exceeds 1e-4 of the maximum sampled
conditional evidence for that method. These thresholds control initial
refinement only; they do not remove any prior region or suppress its error.
Refine overlapping tree branches until their widths are no larger than the
observed feature bracket. Record all selected brackets before new solves.

Retain positive nested Simpson 3/5 outer rules and positive Gauss–Legendre 3/5
inner rules, explicit support/CDF cuts and propagated inner absolute errors.
Use verified existing inner partitions when tightening tolerances instead of
repeating mode/root searches. For new columns retain the complete 33-point
survey and original discovery procedure. Batch only already-required survey
and quadrature nodes; direct solver values are unchanged. A shared process
pool serves independent columns. No interpolated forward likelihood is used.

[QUADPACK's breakpoint interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
supports targeting difficult integration features, but its errors are estimates.
[SciPy's vector quadrature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad_vec.html)
documents joint vector error control, caching and parallel subdivision. F7
retains its own positive rules and explicit inner-error propagation. Finite
surveys cannot prove that all arbitrarily narrow modes/support islands exist
or have been located. Analytic tests must exercise the complete outer method,
including curved ridges and CDF transitions, before direct evaluations begin.

## Unchanged acceptance gates

Require every evidence, normalized first/second moment and all three CDF
numerator errors, divided by each method's joint evidence, <=0.02 then <=0.005.
Inner tolerances remain one tenth of the outer target. Force every outer
panel to bisect between tolerance stages. Both methods must show successive
mean changes <=0.1 final SD, relative 90%-width changes <=0.1, and each CDF
change <=0.02. Positive-node marginal widths remain numerical approximations.
Use the predefined 0.00125 third target only if the successive-summary gate
fails after 0.005. A passing integral at one tolerance alone is insufficient.

F7 allows up to 80,000 new level-1 solver submissions and 1,024 per column;
the inner 256-panel cap is unchanged. This is a separately recorded budget
for a revised algorithm and two mandatory tolerance stages, not a relabeling
of F6's 8,000-trajectory failure. A durable ledger reserves work before
submission, conservatively counting interrupted/unfinished requests. Record
both reserved submissions and completed evaluations. Caps are failures, not
convergence. An exclusive driver lock prevents duplicate concurrent runs.

After integration stops, choose five solver peaks through outer posterior-mass
quantiles 0.1/0.3/0.5/0.7/0.9 (deduplicated) and record selection before any new
level-2 evaluations. Reuse exact matching verified F6 level-2 checks; otherwise
run level 2. Require no survival disagreement and maximum profile change
<=0.00434 dex. Independently reconstruct quadrature/error/CDF arithmetic,
positive full-prior weights, direct likelihoods and frozen hashes. Only a
passing numerical and temporal result supports wider validation; it does not
by itself establish population calibration or justify replacing production.

Preflight budget correction, before any F7 solver submissions: nearest-column
costs project about 27,431 new trajectories for the two mandatory levels and
61,642 including the optional third level, before extra adaptive refinement.
The initially prepared 24,000 budget was therefore inadequate. Its unused
configuration and protocol are archived with `preflight_24000` names. The final
80,000 ceiling covers the prescribed validation; stop immediately on success.

---

*Publication note.* Sentences recording how this study was authorised and scheduled have been reworded for publication; nothing the protocol commits to has changed. The experiment records name the frozen text by its SHA-256, `ff839aba834ccbdb35418941e29adf05a6513324a3e2355b80e0619336311f41`.
