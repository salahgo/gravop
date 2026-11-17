# Track B preregistration: disjoint-system transfer of the quadrature rule (2026-11-06)

Frozen before any production trajectory of the new systems exists. Plan of record:
`WORKPLAN_V12.md`, Track B (sections B0-B5); this document fixes its free choices and resolves
its ambiguities. Campaign `results/track_b_20261106`, unit `sidm-track-b-20261106.service`, code
`scripts/d1_systems.py` (plugin), `scripts/d1_campaign.py`, tests `tests/test_d1_disjoint.py`.
Unchanged and only imported: `src/statecond/*`, `sc_gen.py`, `sc_experiment.py`, `sc_timehead.py`,
`sc_decomposition.py` (F9 pins some of these; none is modified).

## Question

The quadrature rule p_hat = sqrt(E_A^2 + F^2) / E_A predicts the converged-clock deployment penalty
p = E_dep / E_A with no fitted coefficient (mean within-system RMS 0.055 dex over the five original
systems; constant 0.155). Its residual is exactly
r = log10(p_hat / p) = 0.5 log10[(1 + g^2)(1 - c) / (g^2 + k^2)], with c the cross-term share,
k = ||e(v~)|| / E_A and g = F / E_A (`d0_identity_check.py`, 9e-17 on the archived points). Does the
rule transfer to equations that share no governing equation, domain/boundary condition, prior or
trajectory with the five, and do (c, k) concentrate so that g alone carries an error bar?

## Systems (as implemented; 128-point storage, 48-point tau grid, observable threshold x100)

| id | tier | equation | parameters (uniform box) | observable A |
|---|---|---|---|---|
| quench | 1 | u_t = u_xx + lam (1-u)^-beta, u=0 at x=0,1 | log10 lam [0.6,1.3], beta [0.5,2], a [0,0.5], skew [-0.3,0.3]; u0 = a sin(pi x)(1+skew sin 2pi x)/(1+|skew|) | 1/(1 - max u) |
| pme | 1 | u_t = 0.01 (u^m)_xx + u^(m+dp), periodic, compact support | m [1.5,3], dp [0.5,1.5], log10 a [0,0.5], w [0.12,0.30]; u0 = a cos^2(pi z/2) for |z|<1, z=(x-1/2)/w | max u (stored log10(1+u)) |
| cgl | 1 | A_t = 0.01 (1+i alpha) A_xx + (1+i beta)|A|^2 A, periodic, two real fields | log10 a [-0.1,0.3], alpha, beta [-0.5,0.5], k [0.5,2.5]; A0 = a exp(k(cos 2pi x - 1)) exp(0.5 i sin 2pi x) | max |A| (stored asinh Re, asinh Im) |
| hj | 1 | u_t = u_xx + |u_x|^q + lam, u=0 at x=0,1 (Chebyshev-clustered grid) | log10 lam [0.8,1.4], q [2.5,3.5], log10 a [-1,-0.4], skew [-0.3,0.3] | max wall |u_x| |
| corr | 2 | rd + drift -v u_x, v = 3 + 6(log_a + 0.3)/0.9 (speed tied to the amplitude that sets t_sing) | rd's box | max u |
| bimodal | 2 | film with L [5.6,7.0] straddling the critical 2 pi, a1 [0.05,0.30] | film's other ranges | 1/min h |

Deviations from WORKPLAN_V12, all decided from solver-only prototype runs (seeds 555_000_000+ and
556_000_000+, disjoint from production; no network trained or evaluated on any new system):
* hj: the plan's unforced u_t = eps u_xx + |u_x|^3 either decays or blows up within 1e-4-1e-2 of the
  diffusion time; a constant forcing lam above the critical value sets an O(0.01-0.3) clock while u
  stays bounded and the gradient still blows up at the wall. q is sampled in [2.5, 3.5].
* Diffusion coefficients (pme 0.01, cgl 0.01) and boxes were set so every prototype draw becomes
  singular within t_end; bimodal's box was narrowed so that a fraction does not rupture (dropped, as
  in the suite) and t_sing is heavy-tailed (prototype: 0.7-25 time units, 1.6 decades).
Label convergence (8 draws each, grid doubled, tolerances /10): median relative t_sing change
quench 1e-6, pme 4e-5, cgl 1e-4, hj 1.4e-3, corr 6e-6, bimodal 1.2e-4 (max 2.8e-3, hj).

## Design (mirrors Section 3.2 of the paper)

* 600 draws per system, `sc_gen.generate` unchanged, seed block 70_000_000 + 100_000 k in the
  order quench, pme, cgl, hj, corr, bimodal; non-singular runs dropped by `statecond.data.build`. A
  system with fewer than 300 singular runs is reported as insufficient and not scored.
* `scd.split` (seed 1234, 80/10/10); normalisation from each new system's own training split.
* `sc_experiment.py` unchanged: FNO, seeds 0-2, 200 epochs, arms A (oracle tau), B (published-recipe
  head), C (state). No hyperparameter is changed for any system. A system whose oracle arm does not
  train (mean nRMSE >= 0.25) is reported as such and not scored.
* Converged clock: `sc_timehead.train_head` unchanged, budgets 200/2000/20000 epochs, seeds 0-2; the
  largest budget is used (rule fixed in advance, as in P20).
* Decomposition: `sc_decomposition.run` semantics, with every norm stored as a per-test-trajectory
  sum so that trajectories can be resampled; seeds combined by `sc_decomposition.aggregate` (RMS of
  norms, mean of the cross term). Gate before any new system is scored: the per-trajectory code
  reproduces `decomposition.json` for the five original FNO points (relative < 1e-4) and
  `sc_decomposition.run` (< 1e-6). Closure |F^2 + e^2 + cross - E_dep^2| / E_dep^2 < 1e-6 or the
  system halts.

## Criteria (fixed now)

Per system s (FNO, converged clock): r_s = log10(p_hat_s / p_s). With one backbone the within-system
RMS is |r_s|. Comparator: the constant p_c = 10^mean(log10 p) over the 15 archived converged-clock
points (R0 of `predictor.json`, log10 p_c = 0.1753), applied unchanged; no-penalty rule p = 1 reported.

1. **Primary (confirmatory).** M = mean over the four Tier-1 systems of |r_s|. Passes iff M <= 0.10
   dex and the 95% interval of (M_constant - M) excludes 0. Uncertainty: B = 4000 trajectory
   bootstraps resampled within each system (the same resample for the three seeds), seed spread
   reported separately.
2. **Mechanistic.** For all 11 points (five original FNO, six new) the residual computed from the
   independently aggregated (c, k, g) equals the measured r to 1e-6 (the closure tolerance).
3. **Adversarial.** corr: r < -0.2 dex and c > 0 (the sign is the prediction). bimodal: per-test-
   trajectory residuals r_i (norms aggregated over seeds) have RMS > 0.2 dex and median |r_i| < half
   that RMS. Each must also lie outside the (c, k) box of criterion 4.
4. **Concentration.** Over nine points (the five original FNO points and four Tier-1 points) the
   ranges of c and of k are each within 1.25x their ranges over the original five FNO points.
   (WORKPLAN_V12 says "ten systems"; including the Tier-2 systems, built to break the envelope,
   would contradict criterion 3, so the envelope is defined without them.)
5. **No false trust.** With (c, k) restricted to the box of the original five FNO points, each
   Tier-1 system's measured r lies inside the interval r(g) spans over that box (r is decreasing in
   c and in k, so the interval is [r(g, c_max, k_max), r(g, c_min, k_min)]).

Controls: permutation null (each Tier-1 system's F paired with another Tier-1 system's E_A; the
mean |r| must not beat the constant); closure control (above). Secondary: per-system table (no
penalty / constant / quadrature / bare), per-trajectory RMS and median residuals, clock curves,
systems with p < 1.05 flagged as having no penalty to predict. The optional power-law sweep is not
run.

## What may not happen

No retraining, refitting, threshold change or system replacement after the first production
trajectory. Every outcome is reported, including insufficient or untrained systems. A Tier-1
failure turns the paper's "untested" transfer into a measured boundary. The paper is changed only
on the author's explicit request, and only to what the frozen criteria support.
