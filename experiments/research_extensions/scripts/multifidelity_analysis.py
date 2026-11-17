#!/usr/bin/env python3
"""Experiment F: particle-number ladder, marker robustness and reference uncertainty.

Runs: published B1, B2 (1e5), B5, B6 (2e5), C1, C2 (1e5), C4, C5 from results/independent_halo_20261107, plus the 5e4 ladder in
multifidelity/nsphere/. C3 repeats C1 (same seed) at half the time step: it is a
paired time-step test, not an independent seed, and is excluded from the
pooling (corrected 2026-11-12; it was pooled before). Markers per run, identical for fluid and particles:
  t_x10      first time the mean density inside 0.3% of the mass exceeds 10x
             its pre-collapse minimum (published definition; the minimum of a
             noisy series is biased low, the first crossing early)
  t_x10_s    the same on the series smoothed with a running mean in time
             (window 1.5% of the halo's reference collapse time)
  t_abs3     first time the smoothed series exceeds 3x the analytic initial
             value (no minimum selection)
Model per halo and marker: ln T_{N,s} = mu_inf + b N^-p + e, e ~ N(0, tau^2)
(seed scatter pooled across N), p in {1/2, 1}, plus the no-trend model; fit by
least squares; mu_inf interval by parametric bootstrap over seeds. Model
weights by AICc (few replicates: reported, not over-interpreted).
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import independent_halo_analysis as IH  # noqa: E402

PUB = os.path.join(ROOT, 'results', 'independent_halo_20261107')
LAD = os.path.join(EXT, 'multifidelity', 'nsphere')
TREF = {'B': 545.0, 'C': 140.0}
RUNS = {'B': ['B1', 'B2', 'B5', 'B6', 'B_N5e4_s11', 'B_N5e4_s12'],
        'C': ['C1', 'C2', 'C4', 'C5', 'C_N5e4_s11', 'C_N5e4_s12']}


def load(run):
    p = os.path.join(PUB, f'nsphere_{run}.npz')
    if not os.path.exists(p):
        p = os.path.join(LAD, run, 'measures.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d['t'], d['mean_rho'][:, 1], int(d['N'])


def rho_ic(h):
    from gravop import solver
    u = IH.halo_units(h)
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    return float(IH.fluid_measures(r, m, rho, p, u['I'])[1][1])


def smooth(t, y, win):
    """running mean of ln y over a time window (uniform grid resampling)."""
    tg = np.arange(t[0], t[-1], min(np.diff(t).min(), win / 5))
    ly = np.interp(tg, t, np.log(y))
    k = max(int(round(win / (tg[1] - tg[0]))), 1)
    return tg, np.exp(uniform_filter1d(ly, k, mode='nearest'))


def markers(t, y, h, ric):
    out = {'t_x10': IH.collapse_markers(t, y)['t_x10']}
    ts, ys = smooth(t, y, 0.015 * TREF[h])
    out['t_x10_s'] = IH.collapse_markers(ts, ys)['t_x10']
    ipk = int(np.argmax(ys))
    j = np.where((ys >= 3 * ric) & (np.arange(len(ys)) <= ipk) & (ts > 0.2 * TREF[h]))[0]
    out['t_abs3'] = float(ts[j[0]]) if len(j) else None
    return out


def fit(Ns, lnT, p):
    """least squares ln T = mu + b N^-p; returns mu, b, rss, k."""
    if p is None:
        mu = lnT.mean()
        return mu, 0.0, float(((lnT - mu) ** 2).sum()), 1
    X = np.stack([np.ones_like(Ns), Ns ** (-p)], 1)
    coef, *_ = np.linalg.lstsq(X, lnT, rcond=None)
    return float(coef[0]), float(coef[1]), float(((lnT - X @ coef) ** 2).sum()), 2


def analyse(h):
    ric = rho_ic(h)
    rows = []
    for r in RUNS[h]:
        d = load(r)
        if d is None:
            continue
        t, y, N = d
        rows.append(dict(run=r, N=N, **markers(t, y, h, ric)))
    res = {'rho_ic_f0.003': ric, 'runs': rows, 'markers': {}}
    rng = np.random.default_rng(1)
    for mk in ('t_x10', 't_x10_s', 't_abs3'):
        R = [(r['N'], r[mk]) for r in rows if r[mk]]
        Ns = np.array([x[0] for x in R], float)
        lnT = np.log([x[1] for x in R])
        levels = sorted(set(Ns))
        per = {int(n): dict(mean=float(np.exp(lnT[Ns == n].mean())), n=int((Ns == n).sum()),
                            sd_ln=float(lnT[Ns == n].std(ddof=1)) if (Ns == n).sum() > 1 else None)
               for n in levels}
        models = {}
        n_obs = len(lnT)
        for name, p in (('none', None), ('p1/2', 0.5), ('p1', 1.0)):
            if p is not None and len(levels) < 2:
                continue
            mu, b, rss, k = fit(Ns, lnT, p)
            dof = n_obs - k
            tau2 = rss / max(dof, 1)
            kv = k + 1   # regression coefficients plus the fitted Gaussian variance
            aicc = n_obs * np.log(rss / n_obs + 1e-300) + 2 * kv + (2 * kv * (kv + 1) / max(n_obs - kv - 1, 1))
            boot = []
            X = np.stack([np.ones_like(Ns), Ns ** (-(p or 0.0))], 1) if p else None
            fitted = (X @ np.array([mu, b])) if p else np.full_like(lnT, mu)
            for _ in range(2000):
                yb = fitted + rng.normal(0, np.sqrt(tau2), len(lnT))
                boot.append(fit(Ns, yb, p)[0])
            lo, hi = np.percentile(boot, [16, 84])
            models[name] = dict(T_inf=float(np.exp(mu)), T_inf_68=[float(np.exp(lo)), float(np.exp(hi))],
                                sigma_ln_inf=float(np.std(boot)), slope=b, tau_seed_ln=float(np.sqrt(tau2)),
                                dof=dof, aicc=float(aicc),
                                shift_2e5_to_inf=float(np.exp(mu - fit(Ns, lnT, p)[0]) if False else
                                                       (np.exp(mu) / np.exp(mu + b * 2e5 ** (-(p or 1))) - 1) if p else 0.0))
        a = np.array([m['aicc'] for m in models.values()])
        w = np.exp(-0.5 * (a - a.min()))
        for (k_, m), wi in zip(models.items(), w / w.sum()):
            m['aicc_weight'] = float(wi)
        res['markers'][mk] = dict(per_N=per, models=models)
    return res


def main():
    out = {h: analyse(h) for h in ('B', 'C')}
    json.dump(out, open(os.path.join(EXT, 'multifidelity', 'ladder_analysis.json'), 'w'), indent=1)
    for h, r in out.items():
        print('halo', h)
        for row in r['runs']:
            print('  ', row)
        for mk, m in r['markers'].items():
            print('  ', mk, {n: round(v['mean'], 1) for n, v in m['per_N'].items()})
            for nm, mm in m['models'].items():
                print('     ', nm, 'T_inf', round(mm['T_inf'], 1), [round(x, 1) for x in mm['T_inf_68']],
                      'tau', round(mm['tau_seed_ln'], 3), 'w', round(mm['aicc_weight'], 2),
                      '2e5->inf', round(100 * mm['shift_2e5_to_inf'], 1), '%')


if __name__ == '__main__':
    main()
