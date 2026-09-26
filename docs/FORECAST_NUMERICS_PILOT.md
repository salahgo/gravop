# F5 integration pilot — fixed after temporal diagnostics, before pilot solves

2026-09-10. Temporal levels eps_t=1e-4 versus 3e-5 pass the targeted screen:
maximum profile difference 0.003778 dex, no support disagreements on 66 selected
nodes. The old eps_t=3e-4 level fails against both refinements. This pilot uses
eps_t=1e-4, dlnrho=0.01, dlnt=0.025 consistently at every reference node. Reuse
only F5 level-1 profiles; never mix F4's coarse profiles into reference weights.

Implement positive nested Simpson 3x3/5x5 tensor rules. Equally spaced Simpson
rules integrate cubics exactly ([SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html));
the repository independently tests tensor cubics, fourth-order convergence and
a thin diagonal Gaussian ridge with an analytic integral. Use raw absolute
embedded differences, without dividing by 15 or claiming a rigorous bound.

Select six existing F4 observation cases, fixed before new solves: per time,
the largest final F4 embedded-error case and the case at median error rank.
No model fitting or population-calibration claim uses these diagnostic cases.
Use the complete original prior and explicit survival support. Begin with 8x8
cells (a 33x33 fine-node global survey), lattice 65536. Refine the cells containing
each case's highest-likelihood F4 survey node until that point lies on its 5x5
stencil; this seeds narrow known regions. F4 profiles only locate these seed
points. Every numerical integrand value comes from a level-1 direct solve.

Cap the reference bank at 4096 unique level-1 trajectories, including reused
F5 diagnostic nodes. Refine up to 16 cells per round, conserving the cap using
at most 56 new nodes per split (neighbors share nodes). Sequential estimated
relative-error targets: 0.02, 0.005, and 0.00125 fallback. Check all six cases
for successive mean changes <=0.1 final SD, relative 90%-width changes <=0.1,
and CDF changes <=0.02 at both truth parameters and the solver-likelihood
quantity. A cap is failure, never convergence. Preserve intermediate weights.

F4 observation vectors are reused for a numerical pilot only. Recompute their
truth likelihood under the level-1 solver for the diagnostic CDF quantity;
this does not turn old noisy observations into fresh level-1 simulations.
Compare frozen candidate predictions with the solver on the same rule, and
check their numerical convergence separately. Do not perform or interpret
population uniformity tests on the six selected cases. Inspect temporal
stability at newly found highest-posterior nodes with level-2 solves before
recommending broad validation. Only a successful numerical pilot would justify
a separately frozen broad study with new continuous-prior draws.

Changes in both rule and number of targeted observations mean this is not a
matched-budget proof of Simpson's superiority over F4. Report node counts,
error estimates, posterior changes and all failed gates. Production, model
weights, F4 results and manuscript remain unchanged.
