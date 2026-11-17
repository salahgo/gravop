#!/usr/bin/env python3
"""Experiment E: frozen probabilistic forecasts of the D1/D2 particle collapse times.

Written and run BEFORE any D1/D2 particle data are measured. Target: the seed
mean of t_x10 (published marker) in dynamical times, (a) two seeds at 2e5
particles, (b) three seeds at 1e5.

ln T ~ Normal(mu, s^2), with
  mu, closure variance  posterior over closure families fitted to A, B, C
                        (learned_closures/family_posterior.json; weights
                        exp(-chi2/2), chi2 = (A/0.03)^2 + (C/0.041)^2, B fitted
                        exactly at its 2e5 reference); families combined with
                        equal prior weight times their grid-averaged likelihood
  B reference           0.093 x |dlnt_D/dlnC| / |dlnt_B/dlnC| (c_sensitivity.json)
  seed scatter          tau = 0.05 / sqrt(n) (B 0.068, C 0.031 pooled)
  N-bias variation      0.03 (2e5 target, references of B and C are 2e5 means)
  1e5 offset            -0.05 +- 0.03 (B -8%, C -3% between 1e5 and 2e5)
Baselines: point predictions of the harmonic and literature closures.
"""
import hashlib
import json
import os
import time

import numpy as np

EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP = json.load(open(os.path.join(EXT, 'learned_closures', 'family_posterior.json')))
CS = json.load(open(os.path.join(EXT, 'learned_closures', 'c_sensitivity.json')))
PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))


def main():
    out = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), haloes={}, families={})
    pts = [p for p in FP['points'] if p['A_lnres'] is not None and p['C_lnres'] is not None
           and p['D1_t_x10_tdyn'] and p['D2_t_x10_tdyn'] and abs(p['B_lnres']) < 0.01]
    fams = sorted(set(p['family'] for p in pts))
    for h in ('D1', 'D2'):
        sB = np.mean([abs(CS[f'alpha{a}|B']) for a in (1.0, 0.669, 0.5)])
        sD = np.mean([abs(CS[f'alpha{a}|{h}']) for a in (1.0, 0.669, 0.5)])
        famstat = {}
        for f in fams:
            P = [p for p in pts if p['family'] == f]
            chi = np.array([(p['A_lnres'] / 0.03) ** 2 + (p['C_lnres'] / 0.041) ** 2 for p in P])
            w = np.exp(-0.5 * (chi - chi.min()))
            lt = np.log([p[f'{h}_t_x10_tdyn'] for p in P])
            m = float(np.sum(w * lt) / w.sum())
            v = float(np.sum(w * (lt - m) ** 2) / w.sum())
            famstat[f] = dict(mean_ln=m, var_ln=v, n=len(P), chi2_min=float(chi.min()),
                              evidence=float(np.mean(np.exp(-0.5 * chi))),
                              best=P[int(np.argmin(chi))]['params'], best_C=P[int(np.argmin(chi))]['C'],
                              t_best=float(np.exp(lt[int(np.argmin(chi))])))
        ev = np.array([famstat[f]['evidence'] for f in fams])
        fw = ev / ev.sum()
        mu = float(np.sum(fw * [famstat[f]['mean_ln'] for f in fams]))
        vcl = float(np.sum(fw * [famstat[f]['var_ln'] + (famstat[f]['mean_ln'] - mu) ** 2 for f in fams]))
        vB = (0.093 * sD / sB) ** 2
        targets = {}
        for tgt, n, off, voff in (('N2e5_mean_of_2', 2, 0.0, 0.03 ** 2), ('N1e5_mean_of_3', 3, -0.05, 0.03 ** 2 + 0.03 ** 2)):
            s = float(np.sqrt(vcl + vB + 0.05 ** 2 / n + voff))
            m_ = mu + off
            targets[tgt] = dict(median=float(np.exp(m_)), sd_ln=s, ci68=[float(np.exp(m_ - s)), float(np.exp(m_ + s))],
                                ci95=[float(np.exp(m_ - 1.96 * s)), float(np.exp(m_ + 1.96 * s))],
                                components_var=dict(closure=vcl, B_reference=vB, seed=0.05 ** 2 / n, N_bias=voff))
        out['haloes'][h] = dict(targets=targets, family_weights=dict(zip(fams, map(float, fw))),
                                baselines={k: v['t_x10_tdyn'] for k, v in PRE['haloes'][h]['predictions'].items()})
        out['families'][h] = famstat
    s = json.dumps(out, indent=1, sort_keys=True)
    out['sha256'] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(out, open(os.path.join(EXT, 'collapse_probability', 'FORECAST_D.json'), 'w'), indent=1)
    for h, d in out['haloes'].items():
        print(h, 'family weights', {k: round(v, 3) for k, v in d['family_weights'].items()})
        for f, st in out['families'][h].items():
            print('   ', f, 'n', st['n'], 'chi2min', round(st['chi2_min'], 2), 'best', st['best'], round(st['best_C'], 3),
                  't_best', round(st['t_best'], 1), 'mean', round(np.exp(st['mean_ln']), 1), 'sd_ln', round(np.sqrt(st['var_ln']), 3))
        for t, v in d['targets'].items():
            print('   ', t, round(v['median'], 1), [round(x, 1) for x in v['ci68']], [round(x, 1) for x in v['ci95']],
                  {k: round(np.sqrt(x), 3) for k, x in v['components_var'].items()})
        print('    baselines', {k: round(v, 1) for k, v in d['baselines'].items()})


if __name__ == '__main__':
    main()
