#!/usr/bin/env python3
"""Refit of the closure with collapse markers processed identically for fluid
and particle runs.

For haloes B, C, D1, D2 the same function computes, from the central-density
series (mean density inside 0.3% of the mass, time in t0), after the same
running-mean smoothing (window 1.5% of the halo's reference time):
  t_x3, t_x10, t_x30   first crossings of 3, 10, 30 times the pre-collapse minimum
  t_abs3               first crossing of 3 times the initial fluid central density
                       (no minimum selection)
for every particle run (independent seeds only; half-time-step repeats
excluded) and for every fluid run on the (alpha, C) grid of closure_scan.py.
B and C fluid trajectories are read from closure_scan_traj.npz; D1 and D2 are
run here on the same grid. Halo A keeps its profile-registration residual from
closure_fit5.py. References: seed mean over all independent runs (no
particle-number trend), uncertainty standard error with a floor of 3% per run,
plus a 3% systematic. Writes learned_closures/marker_refit.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]
import closure_fit5 as F  # noqa: E402
import independent_halo_analysis as IH  # noqa: E402

PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))['haloes']
for _h in ('D1', 'D2'):
    IH.HALO[_h] = dict(mass=PRE[_h]['mass'], rs=PRE[_h]['rs'], falloff=PRE[_h]['falloff'],
                       kappa=PRE[_h]['kappa'], tfinal_tdyn=1.0)
PUB = os.path.join(ROOT, 'results', 'independent_halo_20261107')
RUNS = {'B': [os.path.join(PUB, f'nsphere_{r}.npz') for r in ('B1', 'B2', 'B5', 'B6')]
             + [os.path.join(EXT, 'multifidelity', 'nsphere', r, 'measures.npz') for r in ('B_N5e4_s11', 'B_N5e4_s12')],
        'C': [os.path.join(PUB, f'nsphere_{r}.npz') for r in ('C1', 'C2', 'C4', 'C5')]
             + [os.path.join(EXT, 'multifidelity', 'nsphere', r, 'measures.npz') for r in ('C_N5e4_s11', 'C_N5e4_s12')]}
for _h in ('D1', 'D2'):
    RUNS[_h] = [os.path.join(EXT, 'closure_design', 'nsphere', f'{_h}_{s}', 'measures.npz')
                for s in ('s21', 's22', 's23', 'N2e5_s24', 'N2e5_s25')]
TREF = {'B': 525.0, 'C': 138.0, 'D1': 66.0, 'D2': 84.0}      # t0, sets the smoothing window only
MARKERS = ('t_x3', 't_x10', 't_x30', 't_abs3')


def ric(h):
    from gravop import solver
    u = IH.halo_units(h)
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    return float(IH.fluid_measures(r, m, rho, p, u['I'])[1][1])


def markers(t, y, h, rho0):
    win = 0.015 * TREF[h]
    tg = np.arange(t[0], t[-1], win / 10)      # fixed step, identical for fluid and particles
    k = max(int(round(win / (tg[1] - tg[0]))), 1)
    ys = np.exp(uniform_filter1d(np.interp(tg, t, np.log(y)), k, mode='nearest'))
    mk = IH.collapse_markers(tg, ys)
    out = {m: mk[m] for m in ('t_x3', 't_x10', 't_x30')}
    ipk = int(np.argmax(ys))
    j = np.where((ys >= 3 * rho0) & (np.arange(len(ys)) <= ipk) & (tg > 0.2 * TREF[h]))[0]
    out['t_abs3'] = float(tg[j[0]]) if len(j) else None
    return out


def fluid_D(args):
    h, a, C, rho0 = args
    from gravop import closure, solver
    u = IH.halo_units(h)
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=a, r=r, m=m, rho=rho, p=p, eps_t=3e-4,
                             timestep_mode='implicit', rho_c_factor=1e4, kn_min=0.02, t_end=400.0, max_snaps=3000)
    mr = np.array([IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I'])[1][1]
                   for i in range(len(run['t']))])
    return f'{h}|{a}|{C}', markers(run['t'], mr, h, rho0)


def main(workers):
    meta = json.load(open(os.path.join(EXT, 'learned_closures', 'closure_scan.json')))['meta']
    AL, CS = meta['alphas'], meta['Cs']
    rho0 = {h: ric(h) for h in RUNS}
    ref = {}
    for h, files in RUNS.items():
        per = [markers(np.load(f)['t'], np.load(f)['mean_rho'][:, 1], h, rho0[h]) for f in files]
        ref[h] = {}
        for m in MARKERS:
            v = np.log([x[m] for x in per if x[m]])
            sd = max(float(v.std(ddof=1)), 0.03)
            ref[h][m] = dict(T=float(np.exp(v.mean())), n=int(len(v)), sd_ln=float(v.std(ddof=1)),
                             sigma=float(np.hypot(sd / np.sqrt(len(v)), 0.03)))
    traj = np.load(os.path.join(EXT, 'learned_closures', 'closure_scan_traj.npz'))
    flu = {}
    for h in ('B', 'C'):
        for a in AL:
            for C in CS:
                k = f'{h}|{a}|{C}'
                if f'{k}|t' in traj.files:
                    flu[k] = markers(traj[f'{k}|t'], traj[f'{k}|mean_rho'][:, 1], h, rho0[h])
    with Pool(workers) as pool:
        flu.update(dict(pool.map(fluid_D, [(h, a, C, rho0[h]) for h in ('D1', 'D2') for a in AL for C in CS],
                                 chunksize=1)))
    GA = F.grid('A')
    fa = np.linspace(0.3, 1.5, 241); fc = np.linspace(np.log(min(CS)), np.log(max(CS)), 261)
    A_, C_ = np.meshgrid(fa, fc, indexing='ij'); P = np.stack([A_.ravel(), C_.ravel()], 1)
    out = {'references': ref, 'fits': {}}
    for m in MARKERS:
        R = {'A': GA(P) / F.REF['A'][1]}
        G = {'A': GA}
        for h in RUNS:
            M = np.full((len(AL), len(CS)), np.nan)
            for i, a in enumerate(AL):
                for j, C in enumerate(CS):
                    v = flu.get(f'{h}|{a}|{C}', {}).get(m)
                    M[i, j] = np.log(v / ref[h][m]['T']) if v else np.nan
            G[h] = RegularGridInterpolator((np.array(AL, float), np.log(CS)), M, bounds_error=False)
            R[h] = G[h](P) / ref[h][m]['sigma']
        chi = sum(R[h] ** 2 for h in R)
        k = int(np.nanargmin(chi))
        ok = np.isfinite(chi) & (chi <= chi[k] + 1)
        res = dict(alpha=float(P[k, 0]), C=float(np.exp(P[k, 1])), chi2=float(chi[k]), dof=len(R) - 2,
                   alpha_range=[float(P[ok, 0].min()), float(P[ok, 0].max())],
                   resid_pct={h: float(100 * np.expm1(G[h](P[k:k + 1])[0])) for h in G})
        for name, (a, C) in (('harmonic', (1.0, 0.712)), ('literature', (0.5, 0.9))):
            p = np.array([[a, np.log(C)]])
            res[name] = dict(chi2=float(sum((G[h](p)[0] / (F.REF['A'][1] if h == 'A' else ref[h][m]['sigma'])) ** 2
                                            for h in G)),
                             resid_pct={h: float(100 * np.expm1(G[h](p)[0])) for h in G})
        for name, a in (('harmonic_Cfit', 1.0), ('alpha_half_Cfit', 0.5)):
            c2 = np.where(np.isclose(P[:, 0], a), chi, np.nan)
            kk = int(np.nanargmin(c2))
            res[name] = dict(C=float(np.exp(P[kk, 1])), chi2=float(c2[kk]))
        out['fits'][m] = res
        print(m, 'alpha', round(res['alpha'], 3), res['alpha_range'], 'C', round(res['C'], 3), 'chi2',
              round(res['chi2'], 2), '| harm', round(res['harmonic']['chi2'], 1), 'Cfit', round(res['harmonic_Cfit']['chi2'], 1),
              '| a=1/2', round(res['literature']['chi2'], 1), 'Cfit', round(res['alpha_half_Cfit']['chi2'], 1),
              {h: round(v, 1) for h, v in res['resid_pct'].items()})
    json.dump(out, open(os.path.join(EXT, 'learned_closures', 'marker_refit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
