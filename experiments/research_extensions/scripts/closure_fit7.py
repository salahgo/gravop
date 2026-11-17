#!/usr/bin/env python3
"""Post-hoc seven-halo closure fit, after the pre-registered test of family E.

Adds haloes E1 and E2 (closure_design/score_E.json; seed mean of three 1e5
runs, sigma_y as pre-registered) to the five haloes of closure_fit5.py. The E
fluid grid (t_x10 in t_dyn on the closure_scan (alpha, C) grid) is computed
here. This fit is NOT a test: E was used to test the frozen five-halo closure
first. Leave-one-halo-out over all seven haloes is reported, with the fits
restricted to the c = 6 haloes (D) or to the c = 10/19 haloes (E) to expose any
dependence on the profile beyond Kn_c.

Writes learned_closures/closure_fit7.json (and closure_scan_E.json, the grid).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]
import closure_fit5 as F  # noqa: E402
import independent_halo_analysis as IH  # noqa: E402

PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_E.json')))['haloes']
for _h, _v in PRE.items():
    IH.HALO[_h] = dict(mass=_v['mass'], rs=_v['rs'], falloff=_v['falloff'], kappa=_v['kappa'], tfinal_tdyn=1.0)


def job(a):
    h, al, C = a
    from gravop import closure, solver
    u = IH.halo_units(h)
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=al, r=r, m=m, rho=rho, p=p, eps_t=3e-4,
                             timestep_mode='implicit', rho_c_factor=1e4, kn_min=0.02, t_end=2000.0, max_snaps=3000)
    mr = np.array([IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I'])[1][1]
                   for i in range(len(run['t']))])
    t = IH.collapse_markers(run['t'], mr)['t_x10']
    return f'{h}|{al}|{C}', None if t is None else t * u['t0_myr'] / u['tdyn_myr']


def main(workers):
    gpath = os.path.join(EXT, 'learned_closures', 'closure_scan_E.json')
    if os.path.exists(gpath):
        SE = json.load(open(gpath))
    else:
        with Pool(workers) as pool:
            SE = dict(pool.map(job, [(h, a, C) for h in ('E1', 'E2') for a in F.AL for C in F.CS], chunksize=1))
        json.dump(SE, open(gpath, 'w'), indent=1)
    sc = json.load(open(os.path.join(EXT, 'closure_design', 'score_E.json')))['haloes']
    refs = dict(F.REF)
    for h in ('E1', 'E2'):
        refs[h] = (sc[h]['T_obs'], sc[h]['sigma_y'])
    G = {h: F.grid(h) for h in F.REF}
    for h in ('E1', 'E2'):
        M = np.full((len(F.AL), len(F.CS)), np.nan)
        for i, a in enumerate(F.AL):
            for j, C in enumerate(F.CS):
                v = SE.get(f'{h}|{a}|{C}')
                M[i, j] = np.log(v / refs[h][0]) if v else np.nan
        G[h] = RegularGridInterpolator((np.array(F.AL, float), np.log(F.CS)), M, bounds_error=False)
    fa = np.linspace(0.3, 1.5, 241); fc = np.linspace(np.log(min(F.CS)), np.log(max(F.CS)), 261)
    A_, C_ = np.meshgrid(fa, fc, indexing='ij'); P = np.stack([A_.ravel(), C_.ravel()], 1)
    LR = {h: G[h](P) for h in G}

    def fit(hs):
        chi = sum((LR[h] / refs[h][1]) ** 2 for h in hs)
        k = int(np.nanargmin(chi))
        ok = np.isfinite(chi) & (chi <= chi[k] + 1)
        return dict(alpha=float(P[k, 0]), C=float(np.exp(P[k, 1])), chi2=float(chi[k]), dof=len(hs) - 2,
                    alpha_range=[float(P[ok, 0].min()), float(P[ok, 0].max())],
                    resid_pct={h: float(100 * np.expm1(LR[h][k])) for h in G}), k

    allh = list(G)
    out = {'refs': refs}
    out['all7'], _ = fit(allh)
    out['loho7'] = {}
    for h in allh:
        _, k = fit([x for x in allh if x != h])
        out['loho7'][h] = float(100 * np.expm1(LR[h][k]))
    out['without_D'], _ = fit([h for h in allh if h not in ('D1', 'D2')])
    out['without_E'], _ = fit([h for h in allh if h not in ('E1', 'E2')])
    for name, (a, C) in (('harmonic', (1.0, 0.712)), ('literature', (0.5, 0.9)), ('fit5', (0.58, 0.85))):
        p = np.array([[a, np.log(C)]])
        out[name] = dict(chi2=float(sum((G[h](p)[0] / refs[h][1]) ** 2 for h in allh)),
                         resid_pct={h: float(100 * np.expm1(G[h](p)[0])) for h in allh})
    for name, a in (('alpha_half_Cfit', 0.5), ('harmonic_Cfit', 1.0)):
        chi = sum((LR[h] / refs[h][1]) ** 2 for h in allh)
        c2 = np.where(np.isclose(P[:, 0], a), chi, np.nan); kk = int(np.nanargmin(c2))
        out[name] = dict(C=float(np.exp(P[kk, 1])), chi2=float(c2[kk]))
    json.dump(out, open(os.path.join(EXT, 'learned_closures', 'closure_fit7.json'), 'w'), indent=1)
    for k in ('all7', 'without_D', 'without_E'):
        r = out[k]
        print(k, round(r['alpha'], 3), r['alpha_range'], round(r['C'], 3), round(r['chi2'], 1), r['dof'],
              {h: round(v, 1) for h, v in r['resid_pct'].items()})
    print('loho7', {h: round(v, 1) for h, v in out['loho7'].items()})
    for k in ('harmonic', 'literature', 'fit5'):
        print(k, round(out[k]['chi2'], 1), {h: round(v, 1) for h, v in out[k]['resid_pct'].items()})
    print('alpha=1/2 Cfit', out['alpha_half_Cfit'], 'harmonic Cfit', out['harmonic_Cfit'])


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
