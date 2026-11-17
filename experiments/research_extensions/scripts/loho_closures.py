#!/usr/bin/env python3
"""Leave-one-halo-out cross-validation of closure families (falsification of
added flexibility). Points from learned_closures/family_posterior.json (C solved
from halo B for every shape, so B is in every fit). Held-out haloes: A, C, D1, D2.
Residuals: A ln s (N-body registration, sigma 0.03 assumed), C ln(t/139.53)
(sigma 0.041), D: ln(T_obs / T_pred) with T_pred offset -0.05 if the observed
target is the 1e5 mean, sigma_y from score_D.json. Fixed baselines: harmonic
(production, C fitted on A) and literature (fitted to nothing here).
"""
import json, os
import numpy as np
EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP = json.load(open(os.path.join(EXT, 'learned_closures', 'family_posterior.json')))['points']
SC = json.load(open(os.path.join(EXT, 'closure_design', 'score_D.json')))['haloes']
TR = json.load(open(os.path.join(os.path.dirname(os.path.dirname(EXT)), 'results', 'closure_20261108', 'transfer.json')))['haloes']
PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))['haloes']


def resid(p):
    r = {'A': (p['A_lnres'], 0.03), 'C': (p['C_lnres'], 0.041)}
    for h in ('D1', 'D2'):
        if h in SC and p.get(f'{h}_t_x10_tdyn'):
            t = SC[h]['target']
            off = 0.0 if t['N'] == 200_000 else -0.05
            r[h] = (float(np.log(p[f'{h}_t_x10_tdyn']) + off - np.log(t['T_obs'])), t['sigma_y'])
    return r


def main():
    pts = [p for p in FP if p['A_lnres'] is not None and p['C_lnres'] is not None and abs(p['B_lnres']) < 0.01]
    haloes = ['A', 'C'] + [h for h in ('D1', 'D2') if h in SC]
    out = {'held_out': haloes, 'families': {}, 'fixed': {}}
    for fam in sorted(set(p['family'] for p in pts)):
        P = [(p, resid(p)) for p in pts if p['family'] == fam]
        loho = {}
        for h in haloes:
            best = min(P, key=lambda pr: sum((pr[1][k][0] / pr[1][k][1]) ** 2 for k in haloes if k != h and k in pr[1]))
            if h in best[1]:
                loho[h] = dict(params=best[0]['params'], C=best[0]['C'], resid_ln=best[1][h][0], z=best[1][h][0] / best[1][h][1])
        allfit = min(P, key=lambda pr: sum((pr[1][k][0] / pr[1][k][1]) ** 2 for k in haloes if k in pr[1]))
        out['families'][fam] = dict(n_points=len(P), loho=loho,
                                    loho_rms_z=float(np.sqrt(np.mean([v['z'] ** 2 for v in loho.values()]))),
                                    loho_rms_resid_pct=float(100 * np.sqrt(np.mean([v['resid_ln'] ** 2 for v in loho.values()]))),
                                    all_fit=dict(params=allfit[0]['params'], C=allfit[0]['C'],
                                                 chi2=float(sum((allfit[1][k][0] / allfit[1][k][1]) ** 2 for k in haloes if k in allfit[1])),
                                                 resid_pct={k: 100 * v[0] for k, v in allfit[1].items()}))
    for name, key, tkey in (('harmonic', 'harmonic', 'harmonic'), ('literature', 'alpha_lit', 'literature')):
        r = {'A': (TR['A'][key]['clock_error'], 0.03), 'C': (np.log(TR['C'][key]['t_x10'] / 139.53), 0.041)}
        for h in ('D1', 'D2'):
            if h in SC:
                t = SC[h]['target']; off = 0.0 if t['N'] == 200_000 else -0.05
                r[h] = (float(np.log(PRE[h]['predictions'][tkey]['t_x10_tdyn']) + off - np.log(t['T_obs'])), t['sigma_y'])
        r['B'] = (TR['B'][key]['clock_error_vs_N2e5_mean'], 0.093)
        out['fixed'][name] = dict(resid_pct={k: 100 * v[0] for k, v in r.items()}, z={k: v[0] / v[1] for k, v in r.items()},
                                  rms_z=float(np.sqrt(np.mean([(v[0] / v[1]) ** 2 for v in r.values()]))))
    json.dump(out, open(os.path.join(EXT, 'learned_closures', 'loho.json'), 'w'), indent=1)
    for f, d in out['families'].items():
        print(f, 'n', d['n_points'], 'LOHO rms z', round(d['loho_rms_z'], 2), 'rms resid %', round(d['loho_rms_resid_pct'], 1),
              {h: (round(v['z'], 2), [round(x, 3) for x in v['params']], round(v['C'], 3)) for h, v in d['loho'].items()})
        print('   all-halo fit', d['all_fit']['params'], round(d['all_fit']['C'], 3), 'chi2', round(d['all_fit']['chi2'], 2),
              {k: round(v, 1) for k, v in d['all_fit']['resid_pct'].items()})
    for n, d in out['fixed'].items():
        print(n, 'rms z', round(d['rms_z'], 2), {k: round(v, 1) for k, v in d['resid_pct'].items()})


if __name__ == '__main__':
    main()
