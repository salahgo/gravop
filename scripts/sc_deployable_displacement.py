#!/usr/bin/env python3
"""E45 -- the displacement without the scored population's clock truth.

WHAT THIS CHECKS
----------------
The quadrature rule p_hat = sqrt(E_A^2 + F^2) / E_A is scored in sample
(`sc_decomposition.py` -> predictor.json) and on the reserved draws
(`e43_prospective_score.py` -> prospective.json). In both, F is computed with
the clock head's REALISED error on each scored trajectory, which needs that
trajectory's true terminal time. The manuscript describes F as computable, for
a given size of clock error, from the solver alone; the realised per-trajectory
error is more than a size. This measures how much the rule's score depends on
that extra information, by recomputing F three ways:

  realised     scored trajectories, each with its own head error (as stored;
               reproduced here as a check on this script)
  val_errors   scored trajectories, with the clock-error DISTRIBUTION of the
               frozen validation split: every scored trajectory is displaced by
               every validation error, and the result pooled
  val_only     validation trajectories and validation errors only: F as a
               property of the system and the clock, fixed before any scored
               case exists
  val_median   scored trajectories displaced by one error SIZE, the median
               absolute validation error, taken with either sign: the reading
               "computable for a given size of clock error" literally

and scoring the frozen rule with each. E_A and E_dep are the stored values; the
operators, clock heads, splits and normalisation are the frozen ones. Nothing
is trained or fitted, so the scores are directly comparable with the stored
ones. The aggregation is the stored one: pooled RMS within a seed, RMS over
seeds, RMS of the log10 residual within a system, mean over systems.

Reads data/sc/<system>.h5, data/sc/reserved/<system>.h5, the converged clock
heads and the stored decomposition and prospective records.
Output: results/statecond/deployable_displacement.json
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import statecond.data as scd                                    # noqa: E402
from sc_experiment import Norm, tensors, load_checkpoint        # noqa: E402

RES = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(RES, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']
BACKBONES = ['fno', 'cno', 'transformer']
SEEDS = 3


def _weights(taus, tq):
    """Linear-interpolation index and weight, clamped as np.interp clamps."""
    idx = np.clip(np.searchsorted(taus, tq, side='right') - 1, 0, len(taus) - 2)
    w = np.clip((tq - taus[idx]) / (taus[idx + 1] - taus[idx]), 0.0, 1.0)
    return idx, w


def mean_sq_displacement(U, ratios, taus):
    """Mean over (trajectory, ratio, tau, channel, x) of the squared displacement.

    U: (runs, tau, C, X) reference fields. ratios: (runs, J) true/predicted
    terminal-time ratios, one row per trajectory; the query for tau is
    tau * ratio, exactly as sc_decomposition and e43 build it.
    """
    tot, cnt = 0.0, 0
    for i in range(U.shape[0]):
        tq = np.clip(taus[None, :] * ratios[i][:, None], 0.0, 1.0)   # (J, tau)
        idx, w = _weights(taus, tq)
        Up = ((1 - w)[..., None, None] * U[i][idx]
              + w[..., None, None] * U[i][idx + 1])                  # (J,tau,C,X)
        tot += float(((Up - U[i][None]) ** 2).sum())
        cnt += Up.size
    return tot / cnt


def _two_sided(r_val):
    """True/predicted ratios for a predicted t_sing off by +-m, m the median."""
    m = float(np.median(np.abs(1.0 / r_val - 1.0)))
    return np.array([1.0 / (1.0 + m), 1.0 / (1.0 - m)])


def head_ratio(head, u0, par, lts, t_sd, idx):
    with torch.no_grad():
        lts_hat = head(u0[idx], par[idx]).squeeze(-1)
    return torch.pow(10.0, (lts[idx] - lts_hat) * t_sd).cpu().numpy()


def displacement(name, scored):
    """F per seed under the three constructions, for one system.

    `scored` is 'test' (the frozen test split of data/sc) or 'reserved'.
    """
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    tr, va, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    u, u0, par, _, _, lts = tensors(d, n, 'cpu')
    U = u.numpy()
    if scored == 'test':
        Us, u0s, pars, ltss, sidx = U[te], u0, par, lts, np.asarray(te)
    else:
        dr = scd.load(os.path.join(DATA, 'reserved', f'{name}.h5'))
        ur, u0s, pars, _, _, ltss = tensors(dr, n, 'cpu')   # frozen Norm
        Us, sidx = ur.numpy(), np.arange(ur.shape[0])
    taus = scd.tau_grid()
    va = np.asarray(va)
    rows = []
    for s in range(SEEDS):
        ck = os.path.join(CKPT, f'{name}_fno_time_conv_seed{s}.pt')
        if not os.path.exists(ck):
            continue
        head, _ = load_checkpoint(ck, 'cpu')
        head.eval()
        r_scored = head_ratio(head, u0s, pars, ltss, n.t_sd, sidx)
        r_val = head_ratio(head, u0, par, lts, n.t_sd, va)
        rows.append({
            'seed': s,
            'realised': np.sqrt(mean_sq_displacement(
                Us, r_scored[:, None], taus)),
            'val_errors': np.sqrt(mean_sq_displacement(
                Us, np.broadcast_to(r_val, (len(Us), len(r_val))), taus)),
            'val_only': np.sqrt(mean_sq_displacement(
                U[va], np.broadcast_to(r_val, (len(va), len(r_val))), taus)),
            'val_median': np.sqrt(mean_sq_displacement(
                Us, np.broadcast_to(_two_sided(r_val), (len(Us), 2)), taus)),
            'clock_err_pct_scored': float(100 * np.median(np.abs(1 / r_scored - 1))),
            'clock_err_pct_val': float(100 * np.median(np.abs(1 / r_val - 1))),
            'n_scored': int(len(Us)), 'n_val': int(len(va)),
        })
    rms = lambda k: float(np.sqrt(np.mean([r[k] ** 2 for r in rows])))
    agg = {k: rms(k) for k in ('realised', 'val_errors', 'val_only', 'val_median')}
    agg['per_seed'] = [{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()} for r in rows]
    return agg


def score(points, F):
    """Mean over systems of the within-system RMS log10 residual."""
    per = {}
    for s in SYSTEMS:
        res = [np.log10(np.hypot(p['E_A'], F[s]) / p['E_A'])
               - np.log10(p['E_dep'] / p['E_A'])
               for k, p in points.items() if k.rsplit('_', 1)[0] == s]
        if res:
            per[s] = float(np.sqrt(np.mean(np.square(res))))
    return {'per_system': per, 'mean_of_system_rms': float(np.mean(list(per.values())))}


def main():
    torch.set_num_threads(2)                  # the population run owns the CPU
    dec = json.load(open(os.path.join(RES, 'decomposition.json')))
    pro = json.load(open(os.path.join(ROOT, 'results', 'e43', 'prospective.json')))
    out = {'protocol': __doc__.split('\n\n')[0].strip(),
           'aggregation': 'pooled RMS per seed, RMS over seeds; RMS log10 '
                          'residual within a system, mean over systems',
           'in_sample': {}, 'reserved': {}}
    for regime, scored, points in (('in_sample', 'test', dec),
                                   ('reserved', 'reserved', pro['points'])):
        pts = {k: v for k, v in points.items()
               if k.rsplit('_', 1)[1] in BACKBONES}
        F = {s: displacement(s, scored) for s in SYSTEMS}
        # the stored F, which `realised` must reproduce
        stored = {s: pts[f'{s}_fno']['F'] for s in SYSTEMS}
        for s in SYSTEMS:
            rel = abs(F[s]['realised'] / stored[s] - 1)
            assert rel < 1e-4, f'{regime} {s}: realised F {F[s]["realised"]} ' \
                               f'does not reproduce the stored {stored[s]}'
        block = {'F': F, 'F_stored': stored}
        for kind in ('realised', 'val_errors', 'val_only', 'val_median'):
            block[kind] = score(pts, {s: F[s][kind] for s in SYSTEMS})
        # the nulls the rule is compared with, as stored for the same points
        if regime == 'in_sample':
            pred = json.load(open(os.path.join(RES, 'predictor.json')))['converged_clock']
            nulls = {'constant': pred['R0_null'], 'no_penalty': pred['R_unit_no_penalty']}
            nulls = {k: {'per_system': {s: v['per_system'][s]['rms_log_residual']
                                        for s in SYSTEMS},
                         'mean_of_system_rms': v['mean_of_system_rms']}
                     for k, v in nulls.items()}
        else:
            nulls = {'constant': pro['rules']['R0_constant'],
                     'no_penalty': pro['rules']['R_unit_no_penalty']}
        block['nulls'] = nulls
        block['against_constant'] = {
            kind: {'factor_on_mean': nulls['constant']['mean_of_system_rms']
                   / block[kind]['mean_of_system_rms'],
                   'systems_behind_constant': sorted(
                       s for s in SYSTEMS if block[kind]['per_system'][s]
                       > nulls['constant']['per_system'][s])}
            for kind in ('realised', 'val_errors', 'val_only', 'val_median')}
        out[regime] = block
        print(f'\n{regime}')
        print(f"  {'system':8s} {'F realised':>11s} {'val errors':>11s} {'val only':>9s} {'val median':>11s}")
        for s in SYSTEMS:
            print(f"  {s:8s} {F[s]['realised']:11.4f} {F[s]['val_errors']:11.4f} "
                  f"{F[s]['val_only']:9.4f} {F[s]['val_median']:11.4f}")
        for kind in ('realised', 'val_errors', 'val_only', 'val_median'):
            b = block[kind]
            print(f"  quadrature score, {kind:10s}: {b['mean_of_system_rms']:.4f}  "
                  + ' '.join(f'{k} {v:.3f}' for k, v in b['per_system'].items()))
        for kind, v in block['against_constant'].items():
            print(f"  vs fitted constant, {kind:10s}: {v['factor_on_mean']:.2f}x on the mean; "
                  f"behind it in {v['systems_behind_constant']}")
    p = os.path.join(RES, 'deployable_displacement.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('\nwrote', os.path.relpath(p, ROOT))


if __name__ == '__main__':
    main()
