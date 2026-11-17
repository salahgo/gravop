#!/usr/bin/env python3
"""Fit R(ln Kn_c) = ln(t_lit/t_harm) on the constant-cross-section design scan,
predict the velocity-dependent and untruncated haloes, report residuals."""
import json, os
import numpy as np
from scipy.interpolate import UnivariateSpline
EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(EXT, 'closure_design', 'design_scan.json')))['results']
from collections import defaultdict
G = defaultdict(dict)
for r in R:
    G[(r['sigma_hat'], r['falloff'])][r['closure']] = r
x, y = [], []
for d in G.values():
    if d['harmonic']['t_x10'] and d['literature']['t_x10']:
        x.append(np.log(d['harmonic']['Kn_c'])); y.append(np.log(d['literature']['t_x10'] / d['harmonic']['t_x10']))
o = np.argsort(x); x, y = np.array(x)[o], np.array(y)[o]
spl = np.polyfit(x, y, 4)
train_rms = float(np.sqrt(np.mean((np.polyval(spl, x) - y) ** 2)))
U = json.load(open(os.path.join(EXT, 'physics_discovery', 'universality_vdep.json')))['results']
H = defaultdict(dict)
for r in U:
    H[(r['sigma0'], r['w'], r['falloff'])][r['closure']] = r
rows = []
for k, d in H.items():
    if d['harmonic']['t_x10'] and d['literature']['t_x10']:
        kn = d['harmonic']['Kn_c']; obs = np.log(d['literature']['t_x10'] / d['harmonic']['t_x10'])
        inside = x.min() <= np.log(kn) <= x.max()
        rows.append(dict(sigma0=k[0], w=k[1], falloff=k[2], Kn_c=kn, R_obs=float(obs),
                         R_pred=float(np.polyval(spl, np.log(kn))), in_range=bool(inside)))
ins = [r for r in rows if r['in_range']]
res = np.array([r['R_obs'] - r['R_pred'] for r in ins]); Robs = np.array([r['R_obs'] for r in ins])
out = dict(train_n=len(x), train_rms=train_rms, test_n=len(ins), test_rms_resid=float(np.sqrt(np.mean(res ** 2))),
           test_rms_R=float(np.sqrt(np.mean(Robs ** 2))), test_max_abs_resid=float(np.abs(res).max()),
           by_w={str(w): float(np.sqrt(np.mean([(r['R_obs'] - r['R_pred']) ** 2 for r in ins if r['w'] == w])))
                 for w in sorted(set(r['w'] for r in ins))},
           untruncated_rms=float(np.sqrt(np.mean([(r['R_obs'] - r['R_pred']) ** 2 for r in ins if r['falloff'] is None]))),
           poly4_coeffs=spl.tolist(), rows=rows)
json.dump(out, open(os.path.join(EXT, 'physics_discovery', 'universality_analysis.json'), 'w'), indent=1)
print({k: v for k, v in out.items() if k not in ('rows', 'poly4_coeffs')})
