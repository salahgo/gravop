# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released version is archived on Zenodo and carries its own version DOI;
the concept DOI always resolves to the most recent one.

## [1.0.0] - 2026-09-26

First public release: this source tree and the archival record of results that
accompanies it.

### Added
- Gravothermal self-interacting dark matter fluid solver with semi-implicit
  conduction, for constant, power-law and velocity-dependent cross-sections,
  cross-validated against a public reference implementation.
- Four reference PDE systems whose evolution ends at a solution-dependent
  event time, with generators, event definitions and convergence studies:
  reaction-diffusion blow-up, Keller-Segel aggregation and thin-film rupture,
  each a finite-time singularity, and viscous Burgers steepening as a regular
  control.
- The rendered figures, in the archival record under `results/figures/`.
- Fourier neural operator emulator, physical-time forecaster, conformal
  calibration and uncertainty-gated execution.
- Bayesian inference pipeline for the cross-section parameters.
- The conditioning-coordinate study: oracle, deployed, state and physical-time
  arms across five systems and three operator backbones, the exact deployed
  error decomposition, and the solver-computable displacement predictor
  scored on reserved parameter draws.
- An acceptance suite of 229 gates that checks the reported results:
  158 verification gates, which recompute their assertion from the solver,
  the data or the model code, and 71 regression gates, which check a
  value against a stored record.
- Snapshot identity before a DOI exists: each tree carries a
  `MANIFEST.sha256`, whose SHA-256 names the tree, and
  `scripts/verify_release.sh` checks a tree against its manifest.
- `setup-env.sh`, a one-command environment bootstrap, and `requirements.lock`,
  the environment that produced the reported results.
- Community and publication metadata: contribution guide, code of conduct,
  licence decisions in `LICENSES.md`, `CITATION.cff`, `.zenodo.json`, issue
  and pull-request templates, and a continuous integration workflow.

### Not yet included
- The results of the continuous-prior population study, which is still
  running. Its frozen specification is in the archival record; its results
  will be a later release.

[1.0.0]: https://github.com/salahgo/gravop/releases/tag/v1.0.0
