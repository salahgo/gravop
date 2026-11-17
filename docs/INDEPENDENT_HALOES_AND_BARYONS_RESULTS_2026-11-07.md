# Independent haloes and static baryonic potentials: results (7 November 2026)

> **Superseded in part (12 November 2026).** The closure verdict and the particle-reference uncertainties below are superseded by [CLOSURE_TEST_RESULTS_2026-11-12.md](CLOSURE_TEST_RESULTS_2026-11-12.md): with particle-number ladders the references of haloes B and C show no resolution trend, and two pre-registered haloes reject the harmonic closure below a central Knudsen number of about one. The static-baryon results stand.

The manuscript tests two extensions of the SIDM application, in Sections 5.5
and 6.4.

1. Calibrate the conductivity on one N-body halo, then predict independent,
   time-resolved haloes with nothing refitted.
2. Add a static baryonic potential, verify it against published results, and
   retrain the emulator for it.

Every number below is read from a result record under
`results/independent_halo_20261107/` or `results/baryons_20261107/`.

## Independent haloes

**Reference code.** NSphere (Kamionkowski & Sigurdson 2025, arXiv:2504.13996)
evolves N particles under the shell-theorem force. SIDM scattering is
simulated by direct Monte Carlo (Kamionkowski, Sigurdson & Slone 2026, PRL 136,
201001).

- Licence: Apache-2.0, <https://github.com/kris-sigurdson/NSphere>, commit `8f14666`.
- It is not redistributed here.
- `scripts/nsphere_reference_runs.py` records the build, the container image
  and every command line.
- Units were checked against the fluid initial conditions: the enclosed mass
  agrees to about 1% outside 0.3 r_s, and the velocity dispersion at r_s to 0.1%.

**Cross-section convention.** Halo A, the isolated run of Fischer, Yu &
Dolag (2025), quotes 80 cm²/g as the viscosity cross-section normalised to the
total for isotropic scattering. NSphere's `--sidm-kappa` is the total isotropic
cross-section, so the calibration transfers without conversion.

**Haloes.** Both use NSphere's truncated NFW profile,
rho ~ 1/[(x + 0.01)(1 + x)²(1 + (x/c)¹⁰)], cut at 85 r_s.

| Halo | M (M☉) | r_s (kpc) | c | σ/m (cm²/g) | σ̂ | Runs |
|---|---|---|---|---|---|---|
| B (published SIDM run of the NSphere authors) | 1.15e9 | 1.18 | 19 | 50 | 0.337 | two seeds |
| C | 1e12 | 25 | 8 | 58.8 | 1.20 | two seeds; half the time step; twice the particles |

**Protocol.**
- The fluid model starts from NSphere's exact initial density, on 300 shells.
- It runs at the conductivity calibrated on halo A: C = 0.712 for the harmonic
  interpolation, and C = 0.878 for the alpha = 1/2 interpolation of van den
  Bosch & Tokayer (2026).
- Nothing is refitted.

**Observables.** These are defined identically for particles and shells:
- the radii enclosing fixed mass fractions;
- the mean density inside 0.3% of the mass (the central density);
- the one-dimensional dispersion inside those radii;
- density profiles in logarithmic bins.

Collapse time is the first time the central density exceeds ten times its
pre-collapse minimum.

**Halo C.**
- Collapse time in NSphere: 135.6 t0, with the two seeds 4% apart.
- Fluid model, harmonic form: collapses 4.9% later. Over the halo body, inside
  0.6 of the truncation radius, its density profiles agree to 0.020 dex RMS,
  its Lagrangian radii to 0.008 dex and its central dispersion to 1.8%.
- Fluid model, alpha = 1/2 form: collapses 11.7% later, with the same
  structural agreement.

Two features of the reference are not used:
- At the truncation edge, about 9 r_s, the fluid model's fixed outer shell
  holds the density up to 0.3 dex above the particles.
- After collapse, the particle core saturates at its resolution and
  re-expands, beyond the Knudsen floor at which the fluid model stops.

**Convergence of the reference (halo C).**
- Half the time step: the collapse time changes by 0.7%.
- Twice the particles: run C4 (seed 3) collapses at 143.7 t0. Run C5 (seed 1)
  collapses at 135.4 t0; C5 is the follow-up campaign's K reference, with the
  same configuration as C4 except the seed. The 10^5-particle seeds collapse at
  133.2 and 138.1 t0.
- The two means differ by 2.9%, no more than the seed scatter (±3%).
- Corrected 2026-11-11: the earlier 5.9% "particle-number shift" compared
  different seeds and was mostly seed scatter. The reference is converged in
  time step; its particle-number dependence is unresolved for halo C, and its
  uncertainty is ±4%.
- Against the 2e5-particle mean (139.5 t0): harmonic +1.9%, alpha = 1/2 fit on
  A +8.5%, literature closure +6.3%.

**Halo B.**
- At 1e5 particles the two seeds collapse at 530 and 478 t0 (11% apart).
- At 2e5 particles (runs B5, B6) they collapse at 569 and 522 t0, mean 545 t0.
  Doubling the particles therefore delays collapse by 8% (6% in halo C), so the
  reference is not converged in particle number. Combining the 2e5 seed scatter
  and the particle-number shift in quadrature gives a reference uncertainty of
  ±9% (±6% for halo C).
- Profiles before collapse agree to 0.023 dex, with Lagrangian radii and
  dispersions within a few per cent, for every closure.

**Conductivity closures** (`src/gravop/closure.py`, `scripts/closure_transfer.py`,
`results/closure_20261108/`). Collapse-time error against N-body (halo A) and
NSphere at 2e5 particles (B, C):

| Closure | Fitted on | A | B | C |
|---|---|---|---|---|
| Harmonic, C = 0.712 (legacy) | halo A | −1.0% | +12.1% | +1.9% |
| alpha = 1/2, C = 0.878 | halo A | 0 | −1.1% | +8.5% |
| alpha = 1/2, C = 0.9 (van den Bosch & Tokayer 2026; literature) | nothing | −2.3% | −3.4% | +6.3% |

- Corrected 2026-11-11: with the second halo-C run at 2e5 particles (seed 1, from the
  follow-up campaign's K reference) the halo-C reference is 139.5 t0 (±4%), not 143.7. The
  literature closure is then 6.3% late on halo C and the harmonic one 1.9%; each closure
  misses one halo by 1.3-1.5 times the reference resolution, so neither is preferred.
- The harmonic closure is 12% late on halo B, beyond its ±9% uncertainty. The
  margin is modest, and because the reference keeps moving with particle
  number, part of the offset may close at higher resolution.
- The central Knudsen number at the largest core is 2.3 (A), 6.7 (B) and
  1.8 (C). The harmonic offset appears in the halo deepest in the
  long-mean-free-path regime.
- At alpha = 1 the module reproduces the production solver and the baryonic
  module bit for bit. At alpha = 1/2 it agrees with the earlier exploratory
  copy to round-off (2e-13).
- The emulator is re-aligned to the literature closure as follows. The
  production draws and seeds are regenerated under it
  (`scripts/gen_dataset_closure.py`, 1704 s, no failures), and five members are
  trained with the production recipe. Their test error is 1.32% against 1.39%
  for production, and their clock error 0.71% against 0.80%. They fit the
  halo-A N-body profiles to 0.031 dex with no calibration to that halo, and
  reproduce their solver on haloes B and C to 0.6% and 0.2%.
**Emulator, zero-shot.**
- Exact truncated profiles lie outside the training families: the ensemble
  spread is 0.11 dex (B) and 0.20 dex (C), 30 to 55 times the in-distribution
  value, and the gate returns both haloes to the solver.
- On the nearest in-family profile (untruncated NFW, same ρ_s and r_s) the
  emulator reproduces the solver's collapse time to 0.3%, with a spread of
  0.004 dex.
- That substitution itself, measured solver against solver, moves the collapse
  by −5% (B) and +19% (C).

## Static baryonic potentials

**Implementation.** `src/gravop/baryons.py`.
- The baryons are a static Hernquist sphere. Following Zhong, Yang & Yu
  (2023), they enter only the hydrostatic equation.
- The initial NFW is in Jeans equilibrium in the total potential, with no
  adiabatic contraction.
- Every other numerical step is imported from `gravop.solver`, unchanged.

**Verification** (`scripts/baryon_verification.py`, `tests/test_baryons.py`).

| Check | Result |
|---|---|
| Bit identity with the solver at M_b = 0 | Constant and Yukawa-Born cross-sections |
| Initial equilibrium: first Newton step | 0.7–1.2% shell motion (DMO baseline 0.43%); 20–33% if baryons are ignored |
| Energy conservation, with ∫Φ_b dm | ≤ 1.3e-4 |
| Grid convergence, compact case | 0.3% (300 → 600 shells); 0.07% (r_min 0.01 → 0.003) |

Collapse times of Zhong et al. (2023), Table 3, at their beta and grid:

| Configuration | Difference |
|---|---|
| Dark matter only | −0.3% |
| Medium baryons | −1.2% |
| Diffuse baryons | −1.6% |
| Compact baryons | −5.2% |
| σ/m = 100 cm²/g | −11.9% |

**Controlled cases on halo A** (`scripts/baryon_cases.py`). Zhong et al.'s
dimensionless baryon structures correspond, for halo A, to 1.8e7 and
3.6e7 M☉ within 0.17 and 0.12 kpc.

| Case | t_end / DMO | Zhong et al. estimate | Largest core | Time of ρ_c minimum |
|---|---|---|---|---|
| DMO | 1 (4.83 Gyr) | – | 0.56 kpc | 12% of the lifetime |
| Extended | 0.584 | 0.585 | 0.18 kpc | 10% |
| Compact | 0.260 | 0.237 | 0.05 kpc | 2.9% |

On a 3×3 grid in mass and size, the analytic estimate agrees for a-hat ≈ 0.1.
It is off by factors of 3 to 4 for a-hat = 0.03.

**Emulator** (`scripts/gen_baryon_dataset.py`, `scripts/train_baryon_emulator.py`).
- Dataset: 1200 trajectories with the D1 cross-section distribution, NFW
  initial profiles, log M̂_b in [−2.5, −1.3] and log â in [−1.5, −0.5]
  (15% without baryons), on 300 shells from r = 0.003 r_s.
- The 81 runs with M̂_b/â > 0.6 are held out from training and model selection.
- The network is unchanged except for two more conditioning inputs, and is
  trained without the physics penalties (their hydrostatic term counts the
  dark-matter mass only).

| Test set | Emulator | Density error | Clock error | Ensemble spread |
|---|---|---|---|---|
| Test, with baryons | Baryon-aware (3 seeds) | 1.4% | 6.7% | 0.007 dex |
| | Production (no baryon input) | 17% | 22% | 0.011 dex |
| Held out (deepest potentials) | Baryon-aware | 8.8% | 26% | 0.030 dex |
| | Production | 44% | 430% | – |

The production emulator's spread barely rises on baryonic haloes, so its gate
does not detect the missing physics.

## Not done

- Halo-B timestep and particle-number tests: about a day each. Convergence is
  tested on halo C.
- A direct numerical comparison with the parametric baryonic model of Yang
  (2024): discussed in the text only.
- Hydrodynamical validation: no public SIDM simulation with baryons exists.
