"""E8 — runtime benchmarks: direct solver vs emulator (GPU and CPU).

Times: (a) solver wall-time per trajectory at representative parameters,
(b) emulator full-trajectory (48 tau x 128 r) latency at batch 1/64/4096,
(c) emulator single-profile latency (likelihood-call shape) at batch 1/1024.
Output: results/e8/bench.json (used for the speedup table).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop import solver, dataset as ds  # noqa: E402
from gravop.model import GravOpNet  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e8')
os.makedirs(OUT, exist_ok=True)


def bench_solver():
    cases = [
        dict(label='const_s0.1', sigma_m=0.1),
        dict(label='const_s0.5', sigma_m=0.5),
        dict(label='yuk_s2_w1', sigma_m=2.0, w=1.0,
             model_lmfp='yukawa_tu', model_smfp='yukawa_tu'),
        dict(label='yuk_s0.3_w0.3', sigma_m=0.3, w=0.3,
             model_lmfp='yukawa_tu', model_smfp='yukawa_tu'),
    ]
    rows = []
    for c in cases:
        label = c.pop('label')
        t0 = time.time()
        res = solver.run_halo(n_shells=300, eps_t=3e-4, rho_c_factor=1e3,
                              kn_min=0.02, t_end=1200.0,
                              timestep_mode='implicit', **c)
        rows.append({'label': label, 'wall_s': time.time() - t0,
                     'status': res['status'], 'n_steps': res['n_steps']})
        print(rows[-1], flush=True)
    return rows


def bench_emulator(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck['args']
    rows = []
    for dev in (['cuda:0', 'cpu'] if torch.cuda.is_available() else ['cpu']):
        model = GravOpNet(n_r=ds.N_R, width=a['width'], modes=a['modes'],
                          n_layers=a['layers']).to(dev).eval()
        model.load_state_dict(ck['model'])
        for n_tau, tag in ((ds.N_TAU, 'trajectory'), (1, 'single_profile')):
            for bs in ((1, 64, 512) if n_tau > 1 else (1, 1024, 16384)):
                B = bs * n_tau
                prof = torch.randn(B, 2, ds.N_R, device=dev)
                cond = torch.rand(B, 4, device=dev)
                with torch.no_grad():
                    for _ in range(3):
                        model(prof, cond)  # warmup
                    if dev != 'cpu':
                        torch.cuda.synchronize()
                    t0 = time.time()
                    n_rep = 20 if B <= 64 * 48 else 5
                    for _ in range(n_rep):
                        model(prof, cond)
                    if dev != 'cpu':
                        torch.cuda.synchronize()
                    dt = (time.time() - t0) / n_rep
                rows.append({'device': dev, 'kind': tag, 'batch': bs,
                             'seconds_per_item': dt / bs,
                             'items_per_second': bs / dt})
                print(rows[-1], flush=True)
    return rows


if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'results', 'runs', 'gravop_s0', 'model.pt')
    out = {'solver': bench_solver(), 'emulator': bench_emulator(ckpt),
           'note': 'speedup = solver wall_s / emulator seconds_per_item '
                   '(trajectory kind)'}
    with open(os.path.join(OUT, 'bench.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('saved bench.json')
