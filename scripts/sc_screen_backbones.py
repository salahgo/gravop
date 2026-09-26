"""Cheap oracle-error screen, to choose which backbones earn a full E22 run.

A full run is 3 seeds x 2 arms x 5 systems plus the sensitivity sweep. This
trains only the tau arm, at a reduced budget, on 2 seeds, and reports the
oracle error alone -- enough to rank architectures before spending the GPU.

Used on 2026-09-02 to pick CNO for P1. At 60 epochs CNO beat the FNO on all
four bespoke systems (rd 0.0974 vs 0.1048, ks 0.0083 vs 0.0102, film 0.0382 vs
0.0428, burgers 0.0096 vs 0.0106) while the transformer won only on rd. Note
the ranking is budget-dependent: at the full 200 epochs CNO's margin on rd
widened from 7% to 32%, so treat this as a screen and not as a result.

Usage: sc_screen_backbones.py [device] [system,system,...] [epochs]
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_experiment as X                                   # noqa: E402
from statecond import data as scd                           # noqa: E402
from statecond.model import BACKBONES                       # noqa: E402

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
systems = sys.argv[2].split(',') if len(sys.argv) > 2 else ['rd', 'ks']
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
for name in systems:
    d = scd.load(os.path.join(ROOT, 'data', 'sc', f'{name}.h5'))
    n_runs, n_tau, n_fields, n_x = d['u'].shape
    n_par = d['par'].shape[1]
    tr, va, te = scd.split(n_runs)
    n = X.Norm(d, tr); T = X.tensors(d, n, dev); u, u0 = T[0], T[1]
    for bk in BACKBONES:
        errs = []
        t0 = time.time()
        for s in (0, 1):
            net, cond, _ = X.train_operator('tau', T, tr, te, s, dev, EPOCHS,
                                            n_fields, n_par, bk)
            errs.append(X.field_err(net, cond, u, u0, te).mean())
        print(f"{name:8s} {bk:12s} A={np.mean(errs):.5f} +-{np.std(errs):.5f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
