"""E9 — Bayesian inference of (sigma0/m, w) from a mock halo density profile.

Setup: truth theta* = (sigma0=2.0, w=1.5), Yukawa-Born model, NFW IC.
Observation: log10 rho at 20 radii in [0.02, 20] r_s at t_obs = 0.7 t_c(theta*),
with 0.0434 dex (10%) Gaussian noise, seed 7.
Posterior A (emulator): emcee, ensemble-mean prediction, 32 walkers x 3000.
Posterior B (solver grid): direct solver on a 31x31 grid over the prior box
  (log10 sigma0 in [-1.5, 1], log10 w in [-1, 1.5]), multiprocessing.
Also: coverage test over 20 emulator-only mocks (argv: coverage).

Outputs: results/e9/{mock.json, samples.npy, grid_logpost.npz, summary.json}
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop import solver, dataset as ds, phase as gphase  # noqa: E402
from gravop.model import GravOpNet  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e9')
os.makedirs(OUT, exist_ok=True)

R_OBS = np.logspace(np.log10(0.02), np.log10(20.0), 20)
NOISE_DEX = 0.0434
MODEL_ERR_INFLATION = 2.0
PRIOR = {'lsig': (-1.5, 1.0), 'lw': (-1.0, 1.5)}
TRUTH = {'sigma0': 2.0, 'w': 1.5}
# must match the training-data generation domain (gen_dataset.SOLVER_KW),
# in particular t_end: the emulator's time normalization is only defined
# within this horizon.
SOLVER_KW = dict(n_shells=300, eps_t=3e-4, rho_c_factor=1e3, kn_min=0.02,
                 t_end=1200.0, timestep_mode='implicit',
                 model_lmfp='yukawa_tu', model_smfp='yukawa_tu')


def solver_profile_at(sigma0, w, t_obs=None, tau=None):
    """rho(R_OBS) from the direct solver; returns (lrho_obs, t_c)."""
    kw = dict(SOLVER_KW)
    if t_obs is not None:
        kw['t_end'] = float(t_obs)
    res = solver.run_halo(sigma_m=sigma0, w=w, **kw)
    # t_norm semantics match the emulator dataset: termination time of the run
    # (rho-collapse or Kn-floor), or t_end for non-collapsing runs.
    tc = res['t_final']
    if tau is not None:
        t_obs = tau * tc
    # interpolate to t_obs
    ts = res['t']
    if t_obs > ts[-1] + 1e-12:  # collapsed before t_obs
        return None, tc
    i = np.clip(np.searchsorted(ts, t_obs) - 1, 0, len(ts) - 2)
    wgt = np.clip((t_obs - ts[i]) / max(ts[i + 1] - ts[i], 1e-300), 0, 1)
    lg = np.log10(R_OBS)
    out = []
    for j in (i, i + 1):
        rm = np.log10(ds.shell_midpoints(res['r'][j]))
        out.append(np.interp(lg, rm, np.log10(res['rho'][j])))
    return (1 - wgt) * out[0] + wgt * out[1], tc


class EnsembleEmulator:
    def __init__(self, ckpt_paths, device='cuda:0'):
        self.models = []
        self.device = device
        for p in ckpt_paths:
            ck = torch.load(p, map_location='cpu', weights_only=False)
            a = ck['args']
            m = GravOpNet(n_r=ds.N_R, width=a['width'], modes=a['modes'],
                          n_layers=a['layers'],
                          cond_dim=a.get('cond_dim', 4)).to(device).eval()
            m.load_state_dict(ck['model'])
            self.models.append(m)
        self.norm = ck['norm']
        self.args = a
        self.cond_kind = a.get('cond_kind', 'tau')
        self.phi_index = a.get('phi_index', 0)
        # NFW initial profile on the fixed grid (same for all evals)
        r, m_, rho, p = solver.nfw_profile(n_shells=300)
        lr0, lv0 = ds.regrid_snapshot(r, rho, p)
        n = self.norm
        prof = np.stack([(lr0 - n['rho_mu']) / n['rho_sd'],
                         (lv0 - n['v_mu']) / n['v_sd']])
        self.prof = torch.tensor(prof, dtype=torch.float32,
                                 device=device)[None]
        lg_grid = np.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R)
        self.interp_x = lg_grid
        # log10 rho at t = 0 on the grid, the reference the phase contrast is
        # measured against; phi = log10 rho(r_phi, t) - lrho0[r_phi]
        self.lrho0 = lr0
        self.r_phi = gphase.phase_radius(self.phi_index)

    def predict_phase(self, lsig, lw, phi, flag):
        """Batched prediction conditioned on a state coordinate.

        No termination time is computed anywhere in here -- that is the whole
        point of the arm. `phi` and `flag` are broadcast against (lsig, lw), so
        a quadrature node over phi costs one extra batched forward pass.
        """
        n = self.norm
        K = len(lsig)
        phi = np.broadcast_to(np.asarray(phi, dtype=float), (K,))
        flag = np.broadcast_to(np.asarray(flag, dtype=float), (K,))
        cond = torch.tensor(np.stack([
            (lsig - n['sig_mu']) / n['sig_sd'],
            (lw - n['w_mu']) / n['w_sd'],
            np.ones(K),
            (phi - n['p_mu']) / n['p_sd'],
            flag], axis=1), dtype=torch.float32, device=self.device)
        prof = self.prof.expand(K, -1, -1)
        preds = []
        with torch.no_grad():
            for m_ in self.models:
                f, _ = m_(prof, cond)
                lrho = f[:, 0].cpu().numpy() * n['rho_sd'] + n['rho_mu']
                preds.append(np.stack([
                    np.interp(np.log10(R_OBS), self.interp_x, row)
                    for row in lrho]))
        preds = np.array(preds)
        return preds.mean(0), preds.std(0)

    def phi_from_observation(self, lrho_obs):
        """Read the state coordinate off an observed profile.

        Returns None when the phase radius lies inside the observation, which
        is the case that matters: the emulator grid starts at 0.0032 r_s but
        E9's mock is sampled from 0.02 r_s outward, so an ensemble trained with
        phi_index = 0 cannot be conditioned on this observation without
        extrapolating. Train that ensemble with --phi-index 23 instead.
        """
        if self.r_phi < R_OBS.min() * (1 - 1e-9):
            return None
        lrho0_at = float(np.interp(np.log10(self.r_phi), self.interp_x,
                                   self.lrho0))
        return float(np.interp(np.log10(self.r_phi), np.log10(R_OBS),
                               lrho_obs) - lrho0_at)

    def predict(self, lsig, lw, t_obs):
        """Batched: lsig, lw arrays (K,). Returns (lrho_obs (K, n_obs) mean,
        std (K, n_obs), tau (K,), valid mask)."""
        n = self.norm
        K = len(lsig)
        cond_base = torch.tensor(np.stack([
            (lsig - n['sig_mu']) / n['sig_sd'],
            (lw - n['w_mu']) / n['w_sd'],
            np.ones(K)], axis=1), dtype=torch.float32, device=self.device)
        prof = self.prof.expand(K, -1, -1)
        preds = []
        with torch.no_grad():
            # first, t_c from each model (tau entry unused by head)
            cond = torch.cat([cond_base, torch.zeros(K, 1,
                                                     device=self.device)], 1)
            tcs = []
            for m_ in self.models:
                _, sc = m_(prof, cond)
                tcs.append(sc[:, 0].cpu().numpy() * n['tc_sd'] + n['tc_mu'])
            ltc = np.mean(tcs, axis=0)
            tau = t_obs / 10 ** ltc
            valid = tau <= 1.0
            tau_c = torch.tensor(np.clip(tau, 0, 1), dtype=torch.float32,
                                 device=self.device)[:, None]
            cond = torch.cat([cond_base, tau_c], 1)
            for m_ in self.models:
                f, _ = m_(prof, cond)
                lrho = (f[:, 0].cpu().numpy() * n['rho_sd'] + n['rho_mu'])
                preds.append(np.stack([
                    np.interp(np.log10(R_OBS), self.interp_x, row)
                    for row in lrho]))
        preds = np.array(preds)  # (M, K, n_obs)
        return preds.mean(0), preds.std(0), tau, valid


_MODEL_ERR = None


def model_err_at_obs():
    """Validation-calibrated per-radius emulator error (dex) at R_OBS."""
    global _MODEL_ERR
    if _MODEL_ERR is None:
        per_r = np.load(os.path.join(OUT, 'model_err_per_r.npy'))
        lg = np.load(os.path.join(OUT, 'model_err_logr.npy'))
        _MODEL_ERR = np.interp(np.log10(R_OBS), lg, per_r)
    return _MODEL_ERR


def log_post_emulator(em, lsig, lw, lrho_obs, t_obs):
    lo = np.array([PRIOR['lsig'][0], PRIOR['lw'][0]])
    hi = np.array([PRIOR['lsig'][1], PRIOR['lw'][1]])
    theta = np.stack([lsig, lw], axis=1)
    inb = np.all((theta >= lo) & (theta <= hi), axis=1)
    lp = np.full(len(lsig), -np.inf)
    if inb.any():
        mu, sd, tau, valid = em.predict(lsig[inb], lw[inb], t_obs)
        var = NOISE_DEX ** 2 + sd ** 2 + (MODEL_ERR_INFLATION * model_err_at_obs()[None, :]) ** 2
        chi2 = np.sum((mu - lrho_obs) ** 2 / var + np.log(var), axis=1)
        vals = -0.5 * chi2
        vals[~valid] = -np.inf
        lp[inb] = vals
    return lp


def log_post_emulator_phase(em, lsig, lw, lrho_obs, phi_obs, like_mask=None,
                            n_quad=1, phi_sigma=NOISE_DEX):
    """Posterior under state-conditioning, with the circularity handled.

    Conditioning on a phi measured from the observation makes the likelihood
    partly self-explaining: the model is told a statistic of the data and then
    scored on data that includes it. Two treatments, both defensible and both
    available here, because they fail in different directions:

    `n_quad > 1` marginalises over phi instead of conditioning on a point
    estimate. phi is a latent state coordinate observed with the same noise as
    the rest of the profile, so the honest likelihood integrates it out --
    Gauss-Hermite over N(phi_obs, phi_sigma). The post-minimum flag is not
    observable from a single epoch either, so it is marginalised too, under a
    flat prior over its two values. This is the correct choice for a
    likelihood and it removes the over-confidence rather than hiding it.

    `like_mask` drops radii from the chi2 -- used to exclude the radii phi was
    read from, so the conditioned quantity is not also fitted.

    The two can be combined; the reported run states which was used.
    """
    lo = np.array([PRIOR['lsig'][0], PRIOR['lw'][0]])
    hi = np.array([PRIOR['lsig'][1], PRIOR['lw'][1]])
    theta = np.stack([lsig, lw], axis=1)
    inb = np.all((theta >= lo) & (theta <= hi), axis=1)
    lp = np.full(len(lsig), -np.inf)
    if not inb.any():
        return lp

    if n_quad > 1:
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
        phis = phi_obs + phi_sigma * nodes
        logw = np.log(weights / weights.sum())
        # the branch flag is not observable from a single epoch either, so it
        # is marginalised under a flat prior over its two values
        flags, flag_logw = (0.0, 1.0), (np.log(0.5), np.log(0.5))
    else:
        phis, logw = np.array([phi_obs]), np.array([0.0])
        # POINT-ESTIMATE ARM: the flag is FIXED to 1 (post-minimum), which is
        # an assumption, not a measurement. It is the right one for E9's mock
        # -- the observation is at tau = 0.7, well into collapse -- but it uses
        # knowledge the deployed setting would not have. The marginalised arm
        # above makes no such assumption, and both arms give the same answer
        # (E29), so nothing in the conclusion rests on this.
        flags, flag_logw = (1.0,), (0.0,)

    m = np.ones(len(R_OBS), dtype=bool) if like_mask is None else like_mask
    terms = []
    for phi, lwt in zip(phis, logw):
        for fl, flwt in zip(flags, flag_logw):
            mu, sd = em.predict_phase(lsig[inb], lw[inb], phi, fl)
            var = (NOISE_DEX ** 2 + sd ** 2
                   + (MODEL_ERR_INFLATION * model_err_at_obs()[None, :]) ** 2)
            chi2 = np.sum((mu[:, m] - lrho_obs[m]) ** 2 / var[:, m]
                          + np.log(var[:, m]), axis=1)
            terms.append(-0.5 * chi2 + lwt + flwt)
    stacked = np.stack(terms)                       # (n_terms, n_inbounds)
    top = stacked.max(axis=0)
    lp[inb] = top + np.log(np.exp(stacked - top).sum(axis=0))
    return lp


def grid_worker(args):
    lsig, lw, t_obs = args
    lrho, tc = solver_profile_at(10 ** lsig, 10 ** lw, t_obs=t_obs)
    return lsig, lw, (None if lrho is None else lrho.tolist()), tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['mock', 'emcee', 'grid', 'coverage',
                                      'summary'])
    ap.add_argument('--workers', type=int, default=22)
    ap.add_argument('--ngrid', type=int, default=31)
    ap.add_argument('--tau_obs', type=float, default=0.7,
                    help='observation epoch as a fraction of t_norm')
    ap.add_argument('--tag', default='',
                    help='suffix for coverage output file')
    ap.add_argument('--conditioning', default='tau', choices=['tau', 'phase'],
                    help='tau uses the predicted termination time (the '
                         'original arm); phase reads the state coordinate off '
                         'the observation (WORKPLAN_V3 P2)')
    ap.add_argument('--runs', default=None,
                    help='ensemble directory prefix; defaults to gravop_s or '
                         'gravop_phase_e9_s to match --conditioning')
    ap.add_argument('--phi-marginalise', type=int, default=1, metavar='N',
                    help='Gauss-Hermite nodes for integrating phi out of the '
                         'likelihood instead of conditioning on a point '
                         'estimate. 1 = condition on the point estimate')
    ap.add_argument('--like-r-min', type=float, default=None,
                    help='drop observed radii below this (in r_s) from the '
                         'likelihood, so the radii phi is read from are not '
                         'also fitted')
    args = ap.parse_args()

    if args.stage == 'mock':
        lrho_true, tc = solver_profile_at(TRUTH['sigma0'], TRUTH['w'],
                                          tau=0.7)
        t_obs = 0.7 * tc
        rng = np.random.default_rng(7)
        lrho_obs = lrho_true + NOISE_DEX * rng.standard_normal(len(R_OBS))
        json.dump({'lrho_true': lrho_true.tolist(),
                   'lrho_obs': lrho_obs.tolist(), 't_obs': t_obs, 'tc': tc,
                   'truth': TRUTH, 'r_obs': R_OBS.tolist(),
                   'noise_dex': NOISE_DEX, 'seed': 7},
                  open(os.path.join(OUT, 'mock.json'), 'w'), indent=1)
        print('mock saved, t_obs =', t_obs, 'tc =', tc)

    elif args.stage == 'emcee':
        import emcee
        mock = json.load(open(os.path.join(OUT, 'mock.json')))
        lrho_obs = np.array(mock['lrho_obs'])
        t_obs = mock['t_obs']
        prefix = args.runs or ('gravop_phase_e9_s'
                               if args.conditioning == 'phase'
                               else 'gravop_s')
        em = EnsembleEmulator([os.path.join(ROOT, 'results', 'runs',
                                            f'{prefix}{s}', 'model.pt')
                               for s in range(5)])
        if args.conditioning == 'tau':
            def lp(theta):
                return log_post_emulator(em, theta[:, 0], theta[:, 1],
                                         lrho_obs, t_obs)
        else:
            phi_obs = em.phi_from_observation(lrho_obs)
            if phi_obs is None:
                try:
                    gphase.check_phase_radius(
                        em.phi_index, float(R_OBS.min()),
                        what="E9's observation", suffix='_e9')
                except ValueError as exc:
                    raise SystemExit(str(exc))
                raise SystemExit('phi could not be read from the observation')
            mask = (None if args.like_r_min is None
                    else R_OBS >= args.like_r_min)
            print(f'phase arm: phi_obs = {phi_obs:+.4f} dex read at '
                  f'r = {em.r_phi:.4f} r_s | quadrature nodes '
                  f'{args.phi_marginalise} | likelihood radii '
                  f'{int(mask.sum()) if mask is not None else len(R_OBS)}'
                  f'/{len(R_OBS)}', flush=True)

            def lp(theta):
                return log_post_emulator_phase(
                    em, theta[:, 0], theta[:, 1], lrho_obs, phi_obs,
                    like_mask=mask, n_quad=args.phi_marginalise)
        rng = np.random.default_rng(11)
        # initialize walkers near the mode found by a cheap emulator grid scan
        g1 = np.linspace(*PRIOR['lsig'], 41)
        g2 = np.linspace(*PRIOR['lw'], 41)
        GG = np.array([(a, b) for a in g1 for b in g2])
        lps = lp(GG)
        best = GG[np.argmax(lps)]
        p0 = best[None, :] + 0.05 * rng.standard_normal((32, 2))
        p0[:, 0] = np.clip(p0[:, 0], *PRIOR['lsig'])
        p0[:, 1] = np.clip(p0[:, 1], *PRIOR['lw'])
        t0 = time.time()
        sam = emcee.EnsembleSampler(32, 2, lp, vectorize=True)
        sam.run_mcmc(p0, 3000, progress=False)
        wall = time.time() - t0
        chain = sam.get_chain(discard=750, flat=True)
        # the tau arm keeps the original filenames so the committed result and
        # everything downstream of it are untouched
        suf = '' if args.conditioning == 'tau' else '_phase'
        np.save(os.path.join(OUT, f'samples{suf}.npy'), chain)
        meta = {'wall_s': wall, 'n_evals': 32 * 3000,
                'acc_frac': float(sam.acceptance_fraction.mean()),
                'mean': chain.mean(0).tolist(),
                'std': chain.std(0).tolist(),
                'conditioning': args.conditioning}
        if args.conditioning == 'phase':
            meta.update({'ensemble_prefix': prefix, 'phi_obs': phi_obs,
                         'phi_radius_rs': em.r_phi,
                         'phi_marginalise_nodes': args.phi_marginalise,
                         'like_r_min': args.like_r_min,
                         'n_likelihood_radii': (int(mask.sum())
                                                if mask is not None
                                                else len(R_OBS))})
        json.dump(meta, open(os.path.join(OUT, f'emcee_meta{suf}.json'), 'w'),
                  indent=1)
        print('emcee done', wall, 's; mean', chain.mean(0), 'std',
              chain.std(0))

    elif args.stage == 'grid':
        mock = json.load(open(os.path.join(OUT, 'mock.json')))
        lrho_obs = np.array(mock['lrho_obs'])
        t_obs = mock['t_obs']
        g1 = np.linspace(*PRIOR['lsig'], args.ngrid)
        g2 = np.linspace(*PRIOR['lw'], args.ngrid)
        tasks = [(a, b, t_obs) for a in g1 for b in g2]
        t0 = time.time()
        with mp.Pool(args.workers) as pool:
            rows = pool.map(grid_worker, tasks)
        logpost = np.full((args.ngrid, args.ngrid), -np.inf)
        for lsig, lw, lrho, tc in rows:
            i = np.argmin(np.abs(g1 - lsig))
            j = np.argmin(np.abs(g2 - lw))
            if lrho is not None:
                chi2 = np.sum((np.array(lrho) - lrho_obs) ** 2 / NOISE_DEX ** 2)
                logpost[i, j] = -0.5 * chi2
        np.savez(os.path.join(OUT, 'grid_logpost.npz'), lsig=g1, lw=g2,
                 logpost=logpost, wall=time.time() - t0)
        print('grid done in', time.time() - t0, 's')

    elif args.stage == 'coverage':
        em = EnsembleEmulator([os.path.join(ROOT, 'results', 'runs',
                                            f'gravop_s{s}', 'model.pt')
                               for s in range(5)])
        import emcee
        rng = np.random.default_rng(99)
        rows = []
        for k in range(20):
            lsig_t = rng.uniform(-1.0, 0.8)
            lw_t = rng.uniform(-0.7, 1.2)
            lrho_true, tc = solver_profile_at(10 ** lsig_t, 10 ** lw_t,
                                              tau=args.tau_obs)
            if lrho_true is None:
                continue
            t_obs = args.tau_obs * tc
            lrho_obs = lrho_true + NOISE_DEX * rng.standard_normal(len(R_OBS))
            def lp(theta):
                return log_post_emulator(em, theta[:, 0], theta[:, 1],
                                         lrho_obs, t_obs)
            g1 = np.linspace(*PRIOR['lsig'], 41)
            g2 = np.linspace(*PRIOR['lw'], 41)
            GG = np.array([(a, b) for a in g1 for b in g2])
            best = GG[np.argmax(lp(GG))]
            p0 = best[None, :] + 0.05 * rng.standard_normal((32, 2))
            p0[:, 0] = np.clip(p0[:, 0], *PRIOR['lsig'])
            p0[:, 1] = np.clip(p0[:, 1], *PRIOR['lw'])
            sam = emcee.EnsembleSampler(32, 2, lp, vectorize=True)
            sam.run_mcmc(p0, 2000, progress=False)
            ch = sam.get_chain(discard=700, flat=True)
            q_sig = float(np.mean(ch[:, 0] < lsig_t))
            q_w = float(np.mean(ch[:, 1] < lw_t))
            rows.append({'lsig_t': lsig_t, 'lw_t': lw_t, 'rank_sig': q_sig,
                         'rank_w': q_w, 'tc': tc})
            print(k, rows[-1], flush=True)
        json.dump(rows, open(os.path.join(
            OUT, f'coverage{args.tag}.json'), 'w'), indent=1)

    elif args.stage == 'summary':
        chain = np.load(os.path.join(OUT, 'samples.npy'))
        g = np.load(os.path.join(OUT, 'grid_logpost.npz'))
        lp = g['logpost'] - g['logpost'].max()
        P = np.exp(lp)
        P /= P.sum()
        msig = (P.sum(1) @ g['lsig'])
        mw = (P.sum(0) @ g['lw'])
        ssig = np.sqrt(P.sum(1) @ (g['lsig'] - msig) ** 2)
        sw = np.sqrt(P.sum(0) @ (g['lw'] - mw) ** 2)
        out = {'emulator_mean': chain.mean(0).tolist(),
               'emulator_std': chain.std(0).tolist(),
               'grid_mean': [float(msig), float(mw)],
               'grid_std': [float(ssig), float(sw)],
               'truth': [float(np.log10(TRUTH['sigma0'])),
                         float(np.log10(TRUTH['w']))]}
        json.dump(out, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
