#!/usr/bin/env python3
"""d ln t_x10 / d ln C for haloes B, C, D1, D2 at three closures (central difference, +-5% in C)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import family_posterior as FP  # noqa
out = {}
for fam, par, C in (('alpha', (1.0,), 0.798), ('alpha', (0.669,), 0.83), ('alpha', (0.5,), 0.868)):
    gx, gf = FP.table(fam, par)
    for h in ('B', 'C', 'D1', 'D2'):
        tp, tm = FP.t_x10(h, gx, gf, C * 1.05), FP.t_x10(h, gx, gf, C / 1.05)
        out[f'{fam}{par[0]}|{h}'] = float(np.log(tp / tm) / (2 * np.log(1.05)))
        print(fam, par, h, round(out[f'{fam}{par[0]}|{h}'], 3), flush=True)
json.dump(out, open(os.path.join(EXT, 'learned_closures', 'c_sensitivity.json'), 'w'), indent=1)
