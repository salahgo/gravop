#!/usr/bin/env python3
"""P38 — a calibrated surrogate-solver gate, simulated end to end.

The gate a user can actually run. At deployment the residual is unknown, so the
rule cannot look at it. What IS available is the ensemble spread and the
conformal threshold, and their product is a certified half-width:

    halfwidth = q * sigma_ens        (covers the truth with probability 1-alpha)

Policy: return the emulator answer when its certified half-width fits inside
the user's error budget; otherwise defer to the solver. Cost and risk are then
both measurable, against measured per-call timings.

Simulated on the frozen test split and on both out-of-distribution sets, so the
gate is exercised where it is supposed to fire. GPU only.
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
torch.set_num_threads(1)

from gravop import dataset as ds                      # noqa: E402
from gravop import conformal as C                     # noqa: E402
from train import load_data, split, Norm, make_tensors  # noqa: E402
from sc_conformal import (load_member, ensemble_predict, trajectory_scores,
                          MEMBERS)                    # noqa: E402


def per_trajectory(resid, sigma, n_tau, stat='max'):
    """Reduce a trajectory's field to one number, two ways.

    'max' certifies the worst point anywhere in the field: 48 snapshots x 128
    radii x 2 channels, so the statistic is a maximum over 12,288 correlated
    elements and its calibrated threshold is correspondingly large.

    'rms' certifies the field's root-mean-square error instead. That is the
    quantity this project reports its accuracy in, so it is the quantity a user
    sets a budget on. Both are valid conformal procedures; they differ in WHAT
    is guaranteed, and the choice is the difference between a gate that pays
    and one that does not.
    """
    r = resid.reshape(resid.shape[0] // n_tau, -1)
    g = sigma.reshape(sigma.shape[0] // n_tau, -1)
    if stat == 'max':
        return r.max(1), g.max(1)
    return (np.sqrt((r ** 2).mean(1)), np.sqrt((g ** 2).mean(1)))


def simulate(worst_resid, worst_sigma, q, budget, c_solver, c_emu, c_gate):
    """One policy sweep at one error budget. Costs are measured seconds."""
    halfwidth = q * worst_sigma
    serve = halfwidth <= budget          # emulator answer is certifiable
    defer = ~serve
    n = serve.size
    # served answers carry the emulator's real error; deferred ones are exact
    violations = int(np.sum(serve & (worst_resid > budget)))
    cost_gated = n * c_gate + int(defer.sum()) * c_solver
    return {
        'budget': float(budget),
        'deferral': float(defer.mean()),
        'violations': violations,
        'violation_rate_of_served': (
            float(violations / max(int(serve.sum()), 1))),
        'wall_gated_s': float(cost_gated),
        'wall_solver_only_s': float(n * c_solver),
        'wall_emulator_only_s': float(n * c_emu),
        'speedup_vs_solver': float(n * c_solver / cost_gated),
        'emulator_only_violations': int(np.sum(worst_resid > budget)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'statecond',
                                                  'gated_deployment.json'))
    args = ap.parse_args()
    device = torch.device(args.device)

    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['lrho']))
    norm = Norm(d, tr)
    tensors = make_tensors(d, norm, device)
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=device)
    n_tau = taus.shape[0]
    nets = [load_member(m, device)[0] for m in MEMBERS]

    mu_v, sd_v, y_v = ensemble_predict(nets, tensors, taus, va, device)
    r_v = np.abs(y_v - mu_v)
    qs = {}
    for stat in ('max', 'rms'):
        wr, wsig = per_trajectory(r_v, sd_v, n_tau, stat)
        qs[stat] = C.conformal_quantile(wr / (wsig + C.EPS), args.alpha)

    bench = json.load(open(os.path.join(ROOT, 'results', 'e8', 'bench.json')))
    c_solver = float(np.median([b['wall_s'] for b in bench['solver']]))
    # The emulator runs on the GPU, so its cost is a GPU cost. An earlier
    # version took the median over every trajectory entry at batch >= 64, a set
    # holding two CUDA timings and two CPU timings; the median of that set is
    # the midpoint between a GPU number and a CPU number, which is the cost of
    # no machine that exists. It overstated the emulator by 21x and the gate
    # with it. Select the device the emulator actually runs on.
    emu = [e for e in bench['emulator'] if e['kind'] == 'trajectory'
           and e['batch'] >= 64 and str(e['device']).startswith('cuda')]
    assert emu, 'no GPU emulator timings in bench.json'
    assert not any(str(e['device']) == 'cpu' for e in emu), 'CPU timing leaked in'
    c_emu = float(np.median([e['seconds_per_item'] for e in emu]))
    c_gate = len(MEMBERS) * c_emu

    # fix the budget ladder once, on the test split
    mu_t0, sd_t0, y_t0 = ensemble_predict(nets, tensors, taus, te, device)
    budget_ladder = {}
    for stat in ('max', 'rms'):
        wr0, _ = per_trajectory(np.abs(y_t0 - mu_t0), sd_t0, n_tau, stat)
        budget_ladder[stat] = np.quantile(wr0, [0.5, 0.75, 0.9, 0.95, 0.99])

    sets = {}
    t0 = time.time()
    for tag, loader in (('test', None), ('ood_ic', 'ood_ic.h5'),
                        ('ood_param', 'ood_param.h5')):
        if loader is None:
            mu, sd, y = ensemble_predict(nets, tensors, taus, te, device)
        else:
            dd = load_data(os.path.join(ROOT, 'data', 'd1', loader))
            tt = make_tensors(dd, norm, device)
            mu, sd, y = ensemble_predict(nets, tt, taus,
                                         np.arange(len(dd['lrho'])), device)
        sets[tag] = {}
        for stat in ('max', 'rms'):
            wr, wsig = per_trajectory(np.abs(y - mu), sd, n_tau, stat)
            # One absolute ladder for every set, fixed from the IN-DISTRIBUTION
            # test split. A user sets one error tolerance and then meets
            # whatever data arrives; re-deriving the budget from each set's own
            # residuals would hand the gate the answer it is supposed to
            # discover, and would make an out-of-distribution set look easy
            # precisely because it is hard.
            budgets = budget_ladder[stat]
            sets[tag][stat] = {
                'n_trajectories': int(wr.size),
                'median_statistic': float(np.median(wr)),
                'sweep': [simulate(wr, wsig, qs[stat], b, c_solver, c_emu,
                                   c_gate) for b in budgets],
            }

    out = {
        'protocol': 'docs/CONFORMAL_PREREGISTRATION.md',
        'alpha': args.alpha,
        'conformal_q': qs,
        'cost_provenance': {
            'emulator': 'results/e8/bench.json, cuda trajectory batch>=64',
            'solver': 'results/e8/bench.json, CPU wall per call',
            'gate': f'{len(MEMBERS)} x emulator; measured independently in '
                    'results/statecond/hybrid_timing.json',
        },
        'costs_seconds': {'solver': c_solver, 'emulator': c_emu,
                          'gate': c_gate},
        # The gate here is the 5-member ensemble, and the served answer is its
        # mean, so the surrogate cost is inside the gate cost rather than added
        # to it. The reference formula would double-count it; it is reported
        # alongside only to show the size of the difference.
        'break_even_deferral': C.break_even_deferral_fused(c_gate / c_solver),
        'break_even_deferral_reference_formula': C.break_even_deferral(
            c_emu / c_solver, c_gate / c_solver),
        'cost_model': 'C_hybrid / C_solver = g + d; g subsumes the surrogate',
        'budget_ladder': {k: [float(x) for x in v]
                          for k, v in budget_ladder.items()},
        'budget_note': ('one absolute ladder for all sets, fixed on the '
                        'in-distribution test split'),
        'sets': sets,
        'wall_seconds': time.time() - t0,
    }
    json.dump(out, open(args.out, 'w'), indent=1)

    print(f"q(max) = {qs['max']:.2f}   q(rms) = {qs['rms']:.2f}   "
          f"break-even deferral {out['break_even_deferral']:.4f}")
    for tag, byst in sets.items():
        for stat, b in byst.items():
            print(f"\n{tag} / {stat}  (n={b['n_trajectories']})")
            print(f"  {'budget':>9s} {'defer':>7s} {'speedup':>9s} "
                  f"{'served viol.':>13s} {'emu-only viol.':>15s}")
            for s in b['sweep']:
                print(f"  {s['budget']:9.4f} {s['deferral']:7.3f} "
                      f"{s['speedup_vs_solver']:8.1f}x "
                      f"{s['violations']:13d} "
                      f"{s['emulator_only_violations']:15d}")
    print('\nwrote', os.path.relpath(args.out, ROOT))


if __name__ == '__main__':
    main()
