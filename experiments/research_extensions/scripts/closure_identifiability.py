#!/usr/bin/env python3
"""Which (alpha, C) closures of the generalised interpolation fit haloes A, B, C?

Reads learned_closures/closure_scan.json. For each alpha, the C each halo
requires (C_req) is found by interpolating ln(observable) in ln C:
  A: clock error s - 1 = 0 against the N-body run (time registration);
  B, C: ln t_x10 = ln T_ref (NSphere reference, default the 2e5 two-seed means).
Joint misfit chi2(alpha, C) = sum_h [ln t_h(alpha, C) - ln T_h]^2 / sigma_h^2 with
sigma_B, sigma_C the reference uncertainties and sigma_A an assumed 3% (the
N-body reference has no replicate here; stated, not measured).
Usage: closure_identifiability.py [--refB T --sigB s --refC T --sigC s --tag x]
"""
import argparse
import json
import os

import numpy as np

EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(EXT, 'learned_closures')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refB', type=float, default=545.38)
    ap.add_argument('--sigB', type=float, default=0.093)
    ap.add_argument('--refC', type=float, default=139.53)
    ap.add_argument('--sigC', type=float, default=0.041)
    ap.add_argument('--sigA', type=float, default=0.03)
    ap.add_argument('--tag', default='ref2e5')
    a = ap.parse_args()
    S = json.load(open(os.path.join(D, 'closure_scan.json')))
    alphas, Cs = S['meta']['alphas'], np.array(S['meta']['Cs'])
    R = S['results']
    ref = {'B': (a.refB, a.sigB), 'C': (a.refC, a.sigC)}
    lnres = {}      # halo -> (alpha, C) array of ln(model / reference)
    for h in ('A', 'B', 'C'):
        M = np.full((len(alphas), len(Cs)), np.nan)
        for i, al in enumerate(alphas):
            for j, C in enumerate(Cs):
                r = R.get(f'{h}|{al}|{C}')
                if r is None:
                    continue
                if h == 'A':
                    if r.get('clock_error') is not None:
                        M[i, j] = np.log(1 + r['clock_error'])
                elif r.get('t_x10'):
                    M[i, j] = np.log(r['t_x10'] / ref[h][0])
        lnres[h] = M
    sig = {'A': a.sigA, 'B': a.sigB, 'C': a.sigC}
    creq = {}
    for h in ('A', 'B', 'C'):
        creq[h] = []
        for i in range(len(alphas)):
            y = lnres[h][i]
            ok = np.isfinite(y)
            x = np.log(Cs[ok])
            yy = y[ok]
            # ln t decreases with ln C; find the zero crossing
            c = None
            for k in range(len(yy) - 1):
                if yy[k] >= 0 >= yy[k + 1] or yy[k] <= 0 <= yy[k + 1]:
                    c = float(np.exp(x[k] - yy[k] * (x[k + 1] - x[k]) / (yy[k + 1] - yy[k])))
                    break
            creq[h].append(c)
    chi2 = sum((lnres[h] / sig[h]) ** 2 for h in ('A', 'B', 'C'))
    i, j = np.unravel_index(np.nanargmin(chi2), chi2.shape)
    # refine on a fine grid by bilinear interpolation of each halo's ln-residual
    from scipy.interpolate import RegularGridInterpolator
    fine_a = np.linspace(min(alphas), max(alphas), 141)
    fine_c = np.exp(np.linspace(np.log(Cs[0]), np.log(Cs[-1]), 261))
    FA, FC = np.meshgrid(fine_a, np.log(fine_c), indexing='ij')
    pts = np.stack([FA.ravel(), FC.ravel()], 1)
    fchi = np.zeros(len(pts))
    fres = {}
    for h in ('A', 'B', 'C'):
        f = RegularGridInterpolator((np.array(alphas), np.log(Cs)), lnres[h], bounds_error=False)
        fres[h] = f(pts)
        fchi += (fres[h] / sig[h]) ** 2
    k = int(np.nanargmin(fchi))
    best = dict(alpha=float(pts[k, 0]), C=float(np.exp(pts[k, 1])), chi2=float(fchi[k]),
                residuals_pct={h: float(100 * (np.exp(fres[h][k]) - 1)) for h in fres})
    # profile chi2 over alpha (min over C)
    prof = []
    for ia, al in enumerate(fine_a):
        sl = slice(ia * len(fine_c), (ia + 1) * len(fine_c))
        kk = int(np.nanargmin(fchi[sl]))
        prof.append(dict(alpha=float(al), C=float(fine_c[kk]), chi2=float(fchi[sl][kk]),
                         res_pct={h: float(100 * (np.exp(fres[h][sl][kk]) - 1)) for h in fres}))
    ok1 = [p for p in prof if p['chi2'] <= best['chi2'] + 1.0]
    out = dict(refs=dict(B=ref['B'], C=ref['C'], sigA_assumed=a.sigA), alphas=alphas, Cs=Cs.tolist(),
               C_required=creq, best_fit=best,
               alpha_1sigma_range=[ok1[0]['alpha'], ok1[-1]['alpha']] if ok1 else None,
               profile=prof[::5],
               note='chi2 with 3 haloes and 2 parameters: 1 degree of freedom; chi2_min > 3.84 rejects the family at 95%')
    json.dump(out, open(os.path.join(D, f'identifiability_{a.tag}.json'), 'w'), indent=1)
    print('C_required per halo:')
    for i_, al in enumerate(alphas):
        print(f'  alpha {al:4}: ' + '  '.join(f"{h} {creq[h][i_] if creq[h][i_] is None else round(creq[h][i_], 3)}" for h in 'ABC'))
    print('best', best)
    print('alpha within dchi2<1:', out['alpha_1sigma_range'])
    for p in prof[::10]:
        print('  ', round(p['alpha'], 3), round(p['C'], 3), round(p['chi2'], 2), {h: round(v, 1) for h, v in p['res_pct'].items()})


if __name__ == '__main__':
    main()
