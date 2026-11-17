#!/usr/bin/env python3
"""Direct solver evaluation of the fitted closure, without grid interpolation.

The five-halo fit (closure_fit5.py) interpolates ln t bilinearly on an (alpha, C)
grid with spacing 0.1 in alpha and 0.05 in C. This script runs the solver at the
best fit and at the ends of the alpha profile interval (C re-optimised on the
grid at each end, as recorded by the fit), with the same code paths as the scans
(closure_scan.job for A, B, C; family_posterior.t_x10 for D1, D2), and compares
the direct residuals with the interpolated ones.

Writes experiments/research_extensions/learned_closures/fit_direct_check.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import closure_fit5 as F  # noqa: E402


def job(a):
    h, al, C = a
    if h in ('D1', 'D2'):
        import family_posterior as FP
        import gfamilies as G
        gx, gf = G.alpha(al)
        return h, al, C, FP.t_x10(h, gx, gf, float(C))
    import closure_scan as CS
    _, _, _, d = CS.job((h, al, C))
    return h, al, C, d.get('clock_error') if h == 'A' else d.get('t_x10')


def main():
    fit = json.load(open(os.path.join(EXT, 'learned_closures', 'closure_fit5.json')))
    a0, c0 = fit['all']['alpha'], fit['all']['C']
    lo, hi = fit['all']['alpha_range_dchi2_1']
    G = {h: F.grid(h) for h in F.REF}
    fa = np.linspace(0.3, 1.5, 241); fc = np.linspace(np.log(min(F.CS)), np.log(max(F.CS)), 261)
    A_, C_ = np.meshgrid(fa, fc, indexing='ij'); P = np.stack([A_.ravel(), C_.ravel()], 1)
    chi = sum((G[h](P) / F.REF[h][1]) ** 2 for h in F.REF)
    pts = {'best': (a0, c0)}
    for name, a in (('alpha_lo', lo), ('alpha_hi', hi)):
        m = np.isclose(P[:, 0], a)
        k = int(np.nanargmin(np.where(m, chi, np.nan)))
        pts[name] = (float(P[k, 0]), float(np.exp(P[k, 1])))
    tasks = [(h, round(a, 4), round(c, 5)) for (a, c) in pts.values() for h in F.REF]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 5) as pool:
        res = pool.map(job, tasks, chunksize=1)
    direct = {(h, a, c): v for h, a, c, v in res}
    out = {}
    for name, (a, c) in pts.items():
        rows, chi_d, chi_i = {}, 0.0, 0.0
        for h in F.REF:
            v = direct[(h, round(a, 4), round(c, 5))]
            lr_d = np.log1p(v) if h == 'A' else np.log(v / F.REF[h][0])
            lr_i = float(G[h](np.array([[a, np.log(c)]]))[0])
            rows[h] = dict(direct_pct=float(100 * np.expm1(lr_d)), interp_pct=float(100 * np.expm1(lr_i)))
            chi_d += (lr_d / F.REF[h][1]) ** 2; chi_i += (lr_i / F.REF[h][1]) ** 2
        out[name] = dict(alpha=a, C=c, chi2_direct=float(chi_d), chi2_interp=float(chi_i), haloes=rows)
        print(name, a, round(c, 4), 'chi2 direct', round(chi_d, 2), 'interp', round(chi_i, 2),
              {h: (round(r['direct_pct'], 2), round(r['interp_pct'], 2)) for h, r in rows.items()})
    json.dump(out, open(os.path.join(EXT, 'learned_closures', 'fit_direct_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
