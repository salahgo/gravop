#!/usr/bin/env python3
"""Final alpha-family fit on all five haloes with ladder-pooled references, and
leave-one-halo-out prediction with (alpha, C) refitted in every fold.

References (ln-normal): A time registration 0 +- 0.03 (assumed); B and C from
the no-trend model of multifidelity/ladder_analysis.json (independent seeds
only; 525.1 and 138.0, SE 2.4% / 1.1%), plus a 3% marker/N systematic; D1 69.7, D2 87.6 (1e5 two-seed means, tau floor 3%, plus
3% systematic). Bilinear interpolation of ln t on the (alpha, ln C) grid.
"""
import json, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(EXT, 'learned_closures', 'closure_scan.json')))
SD = json.load(open(os.path.join(EXT, 'learned_closures', 'closure_scan_D.json')))
SC = json.load(open(os.path.join(EXT, 'closure_design', 'score_D.json')))['haloes']
AL, CS = S['meta']['alphas'], np.array(S['meta']['Cs'])
LAD = json.load(open(os.path.join(EXT, 'multifidelity', 'ladder_analysis.json')))
REF = {'A': (1.0, 0.03)}
for _h in ('B', 'C'):   # no-trend ladder estimate, independent seeds only
    _m = LAD[_h]['markers']['t_x10']['models']['none']
    REF[_h] = (_m['T_inf'], float(np.hypot(_m['sigma_ln_inf'], 0.03)))
for h in ('D1', 'D2'):
    t = SC[h]['target']
    REF[h] = (t['T_obs'], float(np.hypot(t['tau'] / np.sqrt(t['n']), 0.03)))


def grid(h):
    M = np.full((len(AL), len(CS)), np.nan)
    for i, a in enumerate(AL):
        for j, C in enumerate(CS):
            if h == 'A':
                r = S['results'].get(f'A|{a}|{C}', {})
                v = r.get('clock_error')
                M[i, j] = np.log(1 + v) if v is not None else np.nan
            elif h in ('B', 'C'):
                v = S['results'].get(f'{h}|{a}|{C}', {}).get('t_x10')
                M[i, j] = np.log(v / REF[h][0]) if v else np.nan
            else:
                v = SD.get(f'{h}|{a}|{C}')
                M[i, j] = np.log(v / REF[h][0]) if v else np.nan
    return RegularGridInterpolator((np.array(AL, float), np.log(CS)), M, bounds_error=False)


def main():
    H = list(REF)
    F = {h: grid(h) for h in H}
    fa = np.linspace(0.3, 1.5, 241); fc = np.log(np.exp(np.linspace(np.log(CS[0]), np.log(CS[-1]), 261)))
    A_, C_ = np.meshgrid(fa, fc, indexing='ij'); pts = np.stack([A_.ravel(), C_.ravel()], 1)
    R = {h: F[h](pts) / REF[h][1] for h in H}

    def best(hs):
        chi = sum(R[h] ** 2 for h in hs)
        k = int(np.nanargmin(chi))
        return k, float(chi[k])
    k, chi = best(H)
    out = dict(refs=REF, all=dict(alpha=float(pts[k, 0]), C=float(np.exp(pts[k, 1])), chi2=chi, dof=len(H) - 2,
                                  resid_pct={h: float(100 * (np.exp(F[h](pts[k:k + 1])[0]) - 1)) for h in H}))
    chi_all = sum(R[h] ** 2 for h in H)
    ok = np.isfinite(chi_all) & (chi_all <= chi + 1)
    out['all']['alpha_range_dchi2_1'] = [float(pts[ok, 0].min()), float(pts[ok, 0].max())]
    out['all']['C_range_dchi2_1'] = [float(np.exp(pts[ok, 1].min())), float(np.exp(pts[ok, 1].max()))]
    out['loho'] = {}
    for h in H:
        kk, _ = best([x for x in H if x != h])
        r = float(F[h](pts[kk:kk + 1])[0])
        out['loho'][h] = dict(alpha=float(pts[kk, 0]), C=float(np.exp(pts[kk, 1])), pred_err_pct=100 * (np.exp(r) - 1),
                              z=r / REF[h][1])
    out['loho_rms_z'] = float(np.sqrt(np.mean([v['z'] ** 2 for v in out['loho'].values()])))
    # fixed closures for reference
    for name, (a, C) in (('harmonic', (1.0, 0.712)), ('literature', (0.5, 0.9))):
        p = np.array([[a, np.log(C)]])
        out[name] = {h: dict(err_pct=float(100 * (np.exp(F[h](p)[0]) - 1)), z=float(F[h](p)[0] / REF[h][1])) for h in H}
        out[name]['chi2'] = float(sum(v['z'] ** 2 for v in out[name].values() if isinstance(v, dict)))
    json.dump(out, open(os.path.join(EXT, 'learned_closures', 'closure_fit5.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
