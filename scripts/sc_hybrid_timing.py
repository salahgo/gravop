#!/usr/bin/env python3
"""P39 -- the gate's own cost, measured rather than assumed.

The gating table prices the hybrid loop as

    C_hybrid / C_solver = g + d,

with g the gate's cost as a share of a solver call. That g was *assumed* to be
five times the per-item emulator cost, on the grounds that the gate is a
five-member ensemble. The assumption skips everything around the forward pass:
host/device transfer, the spread reduction, the threshold comparison and the
Python that joins them. This script deletes the assumption and measures the
whole gate path on a real workload with one wall clock.

It measures two regimes, because they are different claims:

  * per-query latency (one trajectory at a time), the honest cost of a gate
    answering a single arriving query;
  * batched throughput, the cost when a sweep hands the gate many queries at
    once. This is NOT parallel scalability and is not reported as such.

What it does NOT do is re-time the solver. F9 owns all 24 CPU threads until
December and the solver is a CPU code, so re-measuring it here would both
perturb F9 and be measured under contention. The solver cost is therefore the
frozen figure from results/e8/bench.json, and the boundary is stated in the
output: the gate side is measured here, the solver side is quoted.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(1)                       # F9 owns the CPU

from gravop import conformal as C                        # noqa: E402
from gravop import dataset as ds                          # noqa: E402
from train import load_data, split, Norm, make_tensors   # noqa: E402
from sc_conformal import load_member, ensemble_predict, MEMBERS  # noqa: E402


def timed_gate(nets, tensors, taus, idx, device, q, budget, chunk):
    """Run the real gate over `idx` and wall-clock every part of it.

    Returns (seconds, deferral fraction). The clock starts before the first
    tensor moves and stops after the last decision is taken, so transfer,
    reduction and dispatch are all inside it.
    """
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    n_tau = taus.shape[0]
    defer = []
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        _, sd, _ = ensemble_predict(nets, tensors, taus, part, device)
        # the deployed statistic: the field's RMS spread, per trajectory. The
        # gate never sees a residual, so only sd is reduced here.
        g = sd.reshape(sd.shape[0] // n_tau, -1)
        wsig = np.sqrt((g ** 2).mean(1))
        defer.append((q * wsig) > budget)
    torch.cuda.synchronize(device)
    el = time.perf_counter() - t0
    d = np.concatenate(defer)
    return el, float(d.mean()), int(d.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    device = torch.device(a.device)

    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['lrho']))
    norm = Norm(d, tr)
    tensors = make_tensors(d, norm, device)
    gate_art = json.load(open(os.path.join(ROOT, 'results', 'statecond',
                                           'gated_deployment.json')))
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=device)
    nets = [load_member(m, device)[0] for m in MEMBERS]

    q = gate_art['conformal_q']['rms']
    ladder = gate_art['budget_ladder']['rms']
    budget = float(ladder[2])                 # the mid rung of the published ladder
    c_solver = gate_art['costs_seconds']['solver']
    c_emu = gate_art['costs_seconds']['emulator']
    g_assumed = gate_art['costs_seconds']['gate']

    te = np.asarray(te)
    # warm the kernels; a first call pays compilation and allocator growth
    timed_gate(nets, tensors, taus, te[:8], device, q, budget, 8)

    out = {'budget': budget, 'q': q, 'n_trajectories': int(te.size),
           'c_solver_seconds_frozen': c_solver,
           'c_emulator_seconds_frozen': c_emu,
           'gate_seconds_assumed_5x_emulator': g_assumed,
           'solver_note': 'quoted from results/e8/bench.json, NOT re-measured: '
                          'F9 owns all 24 CPU threads',
           'regimes': {}}

    for tag, chunk in (('per_query_latency', 1), ('batched', 64)):
        runs = [timed_gate(nets, tensors, taus, te, device, q, budget, chunk)
                for _ in range(a.repeats)]
        el = float(np.median([r[0] for r in runs]))
        defer, n = runs[0][1], runs[0][2]
        g_meas = el / n                                # seconds per trajectory
        g_rel, gr_assumed = g_meas / c_solver, g_assumed / c_solver
        out['regimes'][tag] = {
            'wall_seconds_all_runs': [r[0] for r in runs],
            'wall_seconds_median': el,
            'gate_seconds_per_trajectory_measured': g_meas,
            'gate_over_assumed': g_meas / g_assumed,
            'deferral': defer,
            'speedup_modelled_with_assumed_g': float(1.0 / (gr_assumed + defer)),
            'speedup_with_measured_g': float(1.0 / (g_rel + defer)),
            'break_even_deferral_measured': C.break_even_deferral_fused(g_rel),
        }
        print(f"{tag:20s} gate {g_meas*1e3:8.3f} ms/traj "
              f"({g_meas/g_assumed:5.2f}x assumed)  defer {defer:.3f}  "
              f"speedup {1.0/(g_rel+defer):6.2f}x "
              f"(modelled {1.0/(gr_assumed+defer):6.2f}x)", flush=True)

    p = os.path.join(ROOT, 'results', 'statecond', 'hybrid_timing.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
