#!/usr/bin/env python3
"""Score the untouched halo family E against its pre-registration, and the halo B
half-time-step control.

Rule (closure_design/PREREGISTRATION_E.json): t_x10 in t_dyn, seed mean of the
three 1e5-particle runs; z = ln(T_obs / T_pred) / sigma_y with
sigma_y = hypot(max(sd_ln, 0.03) / sqrt(n), 0.03); the frozen closure passes on a
halo if |z| <= 2. Halo B control: t_x10 of B_dt2_s1 against B1 (same seed and
particle number, full time step).

Writes closure_design/score_E.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]
import independent_halo_analysis as IH  # noqa: E402

PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_E.json')))


def tx10_tdyn(path):
    d = np.load(path)
    t = d['t'] * float(d['t0_myr']) / float(d['tdyn_myr'])
    return IH.collapse_markers(t, d['mean_rho'][:, 1])['t_x10']


def main():
    out = {'rule': PRE['rule'], 'haloes': {}}
    for h, v in PRE['haloes'].items():
        runs = {}
        for s in (31, 32, 33):
            p = os.path.join(EXT, 'closure_design', 'nsphere', f'{h}_s{s}', 'measures.npz')
            if os.path.exists(p):
                runs[f'{h}_s{s}'] = tx10_tdyn(p)
        if not runs:
            continue
        lv = np.log([x for x in runs.values() if x])
        n = len(lv)
        sd = float(lv.std(ddof=1)) if n > 1 else 0.0
        sig = float(np.hypot(max(sd, 0.03) / np.sqrt(n), 0.03))
        T = float(np.exp(lv.mean()))
        res = dict(runs=runs, n=n, T_obs=T, sd_ln=sd, sigma_y=sig, closures={})
        for c, pdx in v['predictions'].items():
            z = float(np.log(T / pdx['t_x10_tdyn']) / sig)
            res['closures'][c] = dict(T_pred=pdx['t_x10_tdyn'], pred_err_pct=float(100 * (pdx['t_x10_tdyn'] / T - 1)),
                                      z=z, within_2sigma=bool(abs(z) <= 2))
        res['frozen_closure_passes'] = res['closures']['frozen_fit']['within_2sigma']
        out['haloes'][h] = res
        print(h, n, round(T, 2), {c: (round(r['pred_err_pct'], 1), round(r['z'], 2)) for c, r in res['closures'].items()})
    pb = os.path.join(EXT, 'multifidelity', 'nsphere', 'B_dt2_s1', 'measures.npz')
    if os.path.exists(pb):
        d2 = np.load(pb)
        d1 = np.load(os.path.join(ROOT, 'results', 'independent_halo_20261107', 'nsphere_B1.npz'))
        t2 = IH.collapse_markers(d2['t'], d2['mean_rho'][:, 1])['t_x10']
        t1 = IH.collapse_markers(d1['t'], d1['mean_rho'][:, 1])['t_x10']
        out['B_half_time_step'] = dict(B1_t_x10_t0=t1, B_dt2_t_x10_t0=t2, rel=t2 / t1 - 1)
        print('B dt/2', t1, t2, t2 / t1 - 1)
    json.dump(out, open(os.path.join(EXT, 'closure_design', 'score_E.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
