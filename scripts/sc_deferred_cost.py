#!/usr/bin/env python3
"""E36 (V5 Major Comment 5) -- is a deferred query more expensive than a served one?

THE ASSUMPTION BEING TESTED
---------------------------
`gated_deployment.json` prices a hybrid run as

    C_hybrid = n * C_gate + n_deferred * C_solver

with ONE representative C_solver for every deferred query. That is the
constant-cost model, and it is only sound if the gate's choice of what to defer
is independent of what a deferral costs. The gate defers when the ensemble
spread is large, and the spread is large where the halo is hard -- which is
plausibly also where the solver takes the most steps. If so the model
understates the hybrid cost.

THE MEASUREMENT COSTS NOTHING TO MAKE
-------------------------------------
`data/d1/train.h5` carries a `wall` column: the solver's wall time for every
trajectory in the set, recorded when the dataset was generated. The 130 test
trajectories therefore already have per-query solver costs and no new solver
run is needed.

Those walls were taken with 24 generation processes sharing 24 threads, so
their ABSOLUTE scale is contention-inflated -- median 8.75 s against the
uncontended benchmark's 1.295 s. Only their SHAPE is used here: each wall is
divided by the set's median and multiplied by the frozen benchmark cost, so the
mean served/deferred cost ratio, which is the quantity the cost model is wrong
or right about, is unaffected by the contention. The rescaling is stated rather
than buried.

Output: results/statecond/deferred_cost.json
"""
import json
import os
import sys

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(1)

from gravop import dataset as ds                                  # noqa: E402
from gravop import conformal as C                                 # noqa: E402
from train import load_data, split, Norm, make_tensors            # noqa: E402
from sc_conformal import load_member, ensemble_predict, MEMBERS   # noqa: E402
from sc_gated_deployment import per_trajectory                    # noqa: E402


def main():
    device = torch.device('cuda:0')
    alpha = 0.10
    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['lrho']))
    norm = Norm(d, tr)
    tensors = make_tensors(d, norm, device)
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=device)
    n_tau = taus.shape[0]
    nets = [load_member(m, device)[0] for m in MEMBERS]

    mu_v, sd_v, y_v = ensemble_predict(nets, tensors, taus, va, device)
    mu_t, sd_t, y_t = ensemble_predict(nets, tensors, taus, te, device)
    r_v, r_t = np.abs(y_v - mu_v), np.abs(y_t - mu_t)

    with h5py.File(os.path.join(ROOT, 'data', 'd1', 'train.h5'), 'r') as f:
        wall = f['wall'][:][np.asarray(te)]
    bench = json.load(open(os.path.join(ROOT, 'results', 'e8', 'bench.json')))
    c_solver = float(np.median([b['wall_s'] for b in bench['solver']]))
    emu = [e for e in bench['emulator'] if e['kind'] == 'trajectory'
           and e['batch'] >= 64 and str(e['device']).startswith('cuda')]
    c_emu = float(np.median([e['seconds_per_item'] for e in emu]))
    c_gate = len(MEMBERS) * c_emu
    cost = wall / np.median(wall) * c_solver          # shape kept, scale reset

    out = {
        'question': 'does the gate preferentially defer expensive queries?',
        'cost_source': 'data/d1/train.h5:wall, the generation-time solver wall '
                       'per trajectory, rescaled to the frozen benchmark median',
        'contention_note': 'generation ran 24-way parallel; the raw median is '
                           f'{float(np.median(wall)):.3f} s against the '
                           f'uncontended benchmark {c_solver:.6f} s. Only the '
                           'shape of the distribution is used.',
        'per_query_cost_s': {'median': c_solver,
                             'min': float(cost.min()), 'max': float(cost.max()),
                             'p90': float(np.percentile(cost, 90)),
                             'dispersion_max_over_median':
                                 float(cost.max() / np.median(cost))},
        'statistics': {},
    }

    for stat in ('max', 'rms'):
        wr_v, wsig_v = per_trajectory(r_v, sd_v, n_tau, stat)
        wr_t, wsig_t = per_trajectory(r_t, sd_t, n_tau, stat)
        q = C.conformal_quantile(wr_v / (wsig_v + C.EPS), alpha)
        half = q * wsig_t
        ladder = np.quantile(wr_t, [0.5, 0.75, 0.9, 0.95, 0.99])
        rho, p = spearmanr(half, cost)
        rows = []
        for b in ladder:
            serve = half <= b
            defer = ~serve
            n = serve.size
            const = n * c_gate + int(defer.sum()) * c_solver
            matched = n * c_gate + float(cost[defer].sum())
            rows.append({
                'budget': float(b),
                'deferral': float(defer.mean()),
                'mean_cost_deferred_s': (float(cost[defer].mean())
                                         if defer.any() else None),
                'mean_cost_served_s': (float(cost[serve].mean())
                                       if serve.any() else None),
                'cost_ratio_deferred_over_served': (
                    float(cost[defer].mean() / cost[serve].mean())
                    if defer.any() and serve.any() else None),
                'speedup_constant_cost': float(n * c_solver / const),
                'speedup_query_matched': float(float(cost.sum()) / matched),
                'optimism_of_constant_model': float(
                    (n * c_solver / const) / (float(cost.sum()) / matched)),
            })
        out['statistics'][stat] = {
            'q': float(q),
            'spearman_halfwidth_vs_solver_cost': float(rho),
            'spearman_p': float(p),
            'sweep': rows,
        }
        print(f'{stat}: rho(halfwidth, solver cost) = {rho:+.3f} (p={p:.3f})')
        for r in rows:
            print(f"   budget {r['budget']:.5f}  defer {r['deferral']:.3f}"
                  f"  cost ratio def/served "
                  f"{r['cost_ratio_deferred_over_served'] if r['cost_ratio_deferred_over_served'] else float('nan'):.3f}"
                  f"  speed-up {r['speedup_constant_cost']:.1f}x const ->"
                  f" {r['speedup_query_matched']:.1f}x matched")

    json.dump(out, open(os.path.join(ROOT, 'results', 'statecond',
                                     'deferred_cost.json'), 'w'), indent=1)
    print('wrote deferred_cost.json')


if __name__ == '__main__':
    main()
