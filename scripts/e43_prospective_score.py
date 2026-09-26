#!/usr/bin/env python3
"""E43c -- score the frozen predictor on the reserved cases. Out of sample.

Protocol: docs/PROSPECTIVE_PREREGISTRATION.md, committed before the reserved
cases were generated, which was before this script was written.

Nothing here is fitted. The rule is

    p_hat = sqrt(E_A^2 + F^2) / E_A          (quadrature, the paper's primary)
    p_hat = F / E_A                          (bare, reported alongside)

with no coefficient, and the target is p = E_dep / E_A. The operators, the
converged clock heads and the normalisations are the frozen ones; only the
evaluation data are new.

THE NORMALISATION IS THE ONE TRAP HERE. The operators were trained under the
`Norm` of the ORIGINAL training split. Rebuilding it from the reserved set would
silently rescale every input and score a different model. It is therefore built
from `data/sc/<name>.h5`'s training split and applied to the reserved data,
never rebuilt.

Output: results/e43/prospective.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.dirname(__file__))
from sc_experiment import Norm, tensors, load_checkpoint             # noqa: E402
import statecond.data as scd                                        # noqa: E402

RES = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(RES, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
RESERVED = os.path.join(DATA, 'reserved')
OUT = os.path.join(ROOT, 'results', 'e43')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']
BACKBONES = ['fno', 'cno', 'transformer']


def fields_at(net, cond_fn, u0, idx, n_tau, dev):
    out = []
    for j in range(n_tau):
        ti = np.full(len(idx), j)
        with torch.no_grad():
            out.append(net(u0[idx], cond_fn(idx, ti)).cpu().numpy())
    return np.stack(out, axis=1)


def evaluate(name, backbone, seeds, dev):
    """E_A, E_dep, F on the reserved set, under the frozen normalisation."""
    d_tr = scd.load(os.path.join(DATA, f'{name}.h5'))
    tr, _, _ = scd.split(d_tr['u'].shape[0])
    n = Norm(d_tr, tr)                       # frozen: from the TRAINING split
    d = scd.load(os.path.join(RESERVED, f'{name}.h5'))
    u, u0, par, phi, flag, lts = tensors(d, n, dev)
    taus = scd.tau_grid()
    n_tau = len(taus)
    taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)
    idx = np.arange(u.shape[0])
    G = u.cpu().numpy()

    rows = []
    for s in range(seeds):
        tau_ck = os.path.join(CKPT, f'{name}_{backbone}_tau_seed{s}.pt')
        head_ck = os.path.join(CKPT, f'{name}_fno_time_conv_seed{s}.pt')
        if not (os.path.exists(tau_ck) and os.path.exists(head_ck)):
            continue
        net, _ = load_checkpoint(tau_ck, dev); net.eval()
        head, _ = load_checkpoint(head_ck, dev); head.eval()
        with torch.no_grad():
            lts_hat = head(u0, par).squeeze(-1)
        # x n.t_sd: both terms are normalised log-time (the V11 defect)
        ratio = torch.pow(10.0, (lts - lts_hat) * n.t_sd).cpu().numpy()
        tq = np.clip(taus[None, :] * ratio[:, None], 0.0, 1.0)

        def cond_true(r, t):
            return torch.cat([par[r], taus_t[t][:, None]], 1)

        def cond_dep(r, t):
            q = torch.tensor(tq[r, t], dtype=torch.float32, device=dev)
            return torch.cat([par[r], q[:, None]], 1)

        Gv = np.stack([np.stack([np.interp(tq[i], taus,
                                           u[i, :, c, x].cpu().numpy())
                                 for x in range(u.shape[3])], -1)
                       for i in range(len(idx)) for c in range(u.shape[2])])
        Gv = Gv.reshape(len(idx), u.shape[2], n_tau, u.shape[3]).transpose(0, 2, 1, 3)
        Ghat_v = fields_at(net, cond_true, u0, idx, n_tau, dev)
        Ghat_dep = fields_at(net, cond_dep, u0, idx, n_tau, dev)
        m = lambda a: float(np.sqrt((a ** 2).mean()))
        rows.append({'seed': s, 'E_A': m(Ghat_v - G), 'E_dep': m(Ghat_dep - G),
                     'F': m(Gv - G),
                     'clock_err_pct': float(100 * np.median(
                         np.abs(1.0 / ratio - 1.0)))})
        del net, head
        torch.cuda.empty_cache()
    if not rows:
        return None
    rms = lambda k: float(np.sqrt(np.mean([r[k] ** 2 for r in rows])))
    return {'E_A': rms('E_A'), 'E_dep': rms('E_dep'), 'F': rms('F'),
            'clock_err_pct': float(np.mean([r['clock_err_pct'] for r in rows])),
            'n_trajectories': int(u.shape[0]), 'per_seed': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    torch.set_num_threads(2)
    pts = {}
    for name in SYSTEMS:
        if not os.path.exists(os.path.join(RESERVED, f'{name}.h5')):
            print(f'{name}: no reserved set, skipped')
            continue
        for bb in BACKBONES:
            r = evaluate(name, bb, a.seeds, a.device)
            if r is None:
                continue
            pts[f'{name}_{bb}'] = r
            print(f"{name:8s} {bb:12s} n={r['n_trajectories']:3d} "
                  f"E_A {r['E_A']:.4f}  E_dep {r['E_dep']:.4f}  F {r['F']:.4f}  "
                  f"clock {r['clock_err_pct']:.3f}%", flush=True)

    # ---- the frozen rules, applied. No fitting anywhere below.
    def score(pred):
        per = {}
        for s in SYSTEMS:
            v = [np.log10(pred(pts[k]) / (pts[k]['E_dep'] / pts[k]['E_A']))
                 for k in pts if k.startswith(s + '_')]
            if v:
                per[s] = float(np.sqrt(np.mean(np.square(v))))
        return per, (float(np.mean(list(per.values()))) if per else None)

    rules = {
        'R1_quadrature': lambda p: np.sqrt(p['E_A'] ** 2 + p['F'] ** 2) / p['E_A'],
        'R2_bare': lambda p: p['F'] / p['E_A'],
        'R_unit_no_penalty': lambda p: 1.0,
    }
    out = {'protocol': 'docs/PROSPECTIVE_PREREGISTRATION.md',
           'status': 'OUT OF SAMPLE: the 320 reserved draws were never used to '
                     'train, tune, select or score anything',
           'n_points': len(pts), 'points': pts, 'rules': {}}
    for nm, f in rules.items():
        per, mean = score(f)
        out['rules'][nm] = {'per_system': per, 'mean_of_system_rms': mean}

    # the constant null is fitted leave-one-system-out on the RESERVED points,
    # which is the same protocol Table 14 uses for it
    per_const = {}
    logs = {s: [np.log10(pts[k]['E_dep'] / pts[k]['E_A'])
                for k in pts if k.startswith(s + '_')] for s in SYSTEMS}
    logs = {s: v for s, v in logs.items() if v}
    for s in logs:
        other = np.concatenate([logs[o] for o in logs if o != s])
        c = float(np.mean(other))
        per_const[s] = float(np.sqrt(np.mean((np.array(logs[s]) - c) ** 2)))
    out['rules']['R0_constant'] = {
        'per_system': per_const,
        'mean_of_system_rms': float(np.mean(list(per_const.values())))}

    ret = json.load(open(os.path.join(RES, 'predictor.json')))
    out['retrospective_for_comparison'] = {
        k: ret['converged_clock'][k]['mean_of_system_rms']
        for k in ret['converged_clock']
        if isinstance(ret['converged_clock'][k], dict)
        and 'mean_of_system_rms' in ret['converged_clock'][k]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'prospective.json'), 'w'), indent=1)

    print(f"\n{'rule':22s} {'prospective':>12s} {'retrospective':>14s}")
    for nm in ('R1_quadrature', 'R2_bare', 'R0_constant', 'R_unit_no_penalty'):
        p = out['rules'][nm]['mean_of_system_rms']
        q = out['retrospective_for_comparison'].get(
            {'R0_constant': 'R0_null'}.get(nm, nm))
        print(f"{nm:22s} {p:12.4f} {q if q is None else round(q,4):>14}")
    print('wrote results/e43/prospective.json')


if __name__ == '__main__':
    main()
