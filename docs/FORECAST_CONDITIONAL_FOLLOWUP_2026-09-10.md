# F6 follow-up: outer features and exact cache reuse

**Final update: F6 stopped at its 8,000-trajectory cap without convergence.**
The completed-run diagnosis is appended below; earlier observations are retained.

This is a development note written while the frozen F6 pilot is running.
It does not amend its protocol, increase its budget, or establish a passing
numerical result. The manuscript and model checkpoints are unchanged.

## Evidence from the saved evaluations

The first complete outer rule has maximum propagated errors 0.07793 for the
solver and 0.13791 for the candidate, against target 0.02. All 13 inner columns
passed. The largest individual panel contribution comes from log-w [-1, 0.25].
Two observed features suggest where a subsequent outer rule could improve:

- At log-w -0.6875 the best sampled solver log likelihood is -439.326;
  at -0.53125 a peak at log-sigma0 0.925585 reaches -12.8629. This is consistent
  with the narrow ridge entering the prior through its upper sigma boundary.
  It is an inference from sampled values, not a located boundary or proof that
  no unsampled mode exists.
- The sampled peak log likelihood at log-w 0.09375 is -12.16485, below the
  truth likelihood of approximately -12.16132; at 0.25 it is -12.06454, above
  that threshold. The likelihood-CDF level set therefore changes in this region
  if the detected branch varies continuously. Its location is not yet root-found.

For a Gaussian conditional profile with center mu(w), width s, and fixed
prior [a,b], its mass is proportional to
`Phi((b-mu)/s) - Phi((a-mu)/s)`. This changes rapidly when the ridge crosses a
prior edge. If the conditional peak exceeds the diagnostic threshold by d,
the excluded central Gaussian mass grows as `2*Phi(sqrt(2*d))-1` for d>0,
starting with a square-root dependence. Smooth conditional densities can thus
produce difficult outer integrands.

[SciPy's QUADPACK interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
accepts explicit breakpoints at difficult features and documents a sharp-feature
example where adaptive integration misses mass. Its reported error is an
estimate. [The nested interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.nquad.html)
supports variable-dependent options and bounds, but its returned error is the
maximum of individual integration estimates. It would not replace F6's explicit
positive-weight propagation of inner errors without further analysis.

## Bounded analytic benchmark

`scripts/benchmark_conditional_outer.py` compares the existing three base
intervals with additional known feature locations. It uses positive nested
Simpson rules, raw coarse/fine differences and a 257-node cap. Reference values
come from stricter QUADPACK integration with the constructed feature locations.
The two functions represent a Gaussian ridge crossing a prior boundary and the
appearance of a likelihood-CDF level set. They are not fitted SIDM references.

At target 0.005:

| Constructed feature | Original nodes | Feature-aware nodes | Original actual relative error | Feature-aware actual relative error |
|---|---:|---:|---:|---:|
| Prior boundary | 33 | 25 | 0.000631 | 0.000000271 |
| Likelihood-CDF appearance | 25 | 17 | 0.0000135 | 0.000162 |

All 16 benchmark runs reached their requested estimated and actual tolerances.
The improvement is not universal: at target 0.0001 the CDF example needs 41
feature-aware nodes versus 37 original nodes. Known feature locations are
provided for free in this benchmark; discovering them with the direct solver
has an additional cost. These results support testing targeted breakpoints,
not a numerical-convergence claim for F6.

Artifacts: `results/forecast_conditional/outer_feature_benchmark.json`,
`logs/f6_outer_benchmark.log`.

## Cache fix prepared and tested

Subdividing the original upper base interval regenerates an inherited coordinate
as 0.8619692602223811 instead of 0.8619692602223812. The exact-byte cache key
would trigger a duplicate column. `cache_roundoff_diagnostic.json` records this
read-only lookahead; it does not merge or relabel existing data.

`src/gravop/dyadic.py` now constructs nodes from fixed base bounds, an integer
descendant index and its depth. Inherited coordinates remain bit-identical,
and shared base endpoints retain their original representation. Three tests
pass, including the observed miss and refinement through depth 17. The frozen
F6 driver does not import this helper. Adopting it requires a separately
recorded driver revision with base-panel identities in its saved state.

After F6 stops, review its final error decomposition before choosing a new
experiment. If outer error remains limiting, combine exact cache reuse with
explicitly located outer features, retain the full prior and both methods'
support, and count discovery solves in the budget. Preserve the current run
and its gates. A new numerical pilot must pass before any wider calibration.

## Completed-run diagnosis

The [final F6 report](FORECAST_CONDITIONAL_RESULTS_2026-09-10.md) records 29
passing inner columns out of 30 attempted. The last complete outer rule has
25 columns and 6,514 new trajectories. Another 1,486 evaluations did not finish
the next outer rule. The final column received only the 106 evaluations left
in the global budget; this was not a 512-evaluation conditional failure.

The largest final error is **the sigma0 CDF numerator**, 0.0339683 for the solver
and 0.0340014 for the candidate, versus 0.02. The dominant panel is log-w
[0.64929235, 1.5]. Its solver conditional sigma0 CDF rises from 0.268931 at
0.64929235 to effectively 1 at the additional completed column 0.75563080.
Thus the two earlier feature hypotheses were incomplete: outer parameter-CDF
transitions must also be handled. Evidence alone would have been misleading;
its relative errors are 0.003979/0.009279 on the last complete rule.

`scripts/benchmark_parameter_cdf.py` adds eight scalar tests of a Gaussian CDF
transition with an exact antiderivative. All requested estimated and actual
tolerances pass. At target 0.005, locating the transition and its shoulders
reduces nodes from 33 to 25 and actual relative error from 0.001000 to 0.000303.
At 0.00125, the informed rule needs 33 versus 41 nodes but has a larger actual
error; both remain within tolerance. This is an illustrative constructed case,
and the cost of discovering a physical transition remains excluded. Results
are in `results/forecast_conditional/outer_parameter_cdf_benchmark.json` and
`logs/f6_parameter_cdf_benchmark.log`. The original 16-run benchmark is preserved.

The predicted cache miss actually consumed **338 new trajectories** at the
one-ULP-shifted coordinate. Exact reuse would recover that work, but this does
not establish that the complete two-stage pilot would then pass.

The next numerical revision should integrate the tested dyadic-node helper
into saved base-panel identities and locate outer parameter-CDF transitions
as well as prior-edge and likelihood-CDF features. It must count discovery work,
preserve both survival supports and the full prior, and retain the existing
error and successive-summary gates. First benchmark the complete revised
driver; do not simply increase this failed experiment's cap.

The five selected level-2 checks passed: maximum direct-profile change
0.000454454 dex, no survival disagreement. Candidate density RMSE to those
stricter profiles is 0.00246404 dex, with a maximum difference 0.00696265 dex.
The provisional posterior TV is 0.14544 and survival-disagreement posterior
mass is negligible. These targeted findings concern the reused diagnostic case;
they do not establish population calibration or justify replacing production.
