"""E34 (V5 Major Comment 1) -- the same predictions under every aggregation.

THE PROBLEM THE REFEREE IDENTIFIED
----------------------------------
Two headline numbers in this paper answer different questions and the paper
never says which:

  * Table `tab:clock` quotes a converged deployment penalty of 0.97-2.10x.
    That is `timehead.json:penalty_converged`, and its metric is the MEAN over
    (trajectory, tau) slots of the per-slot RMSE -- a mean of RMS values.
  * Table `tab:decomposition` and the predictor study quote E_dep/E_A, which
    from `decomposition.json` spans 1.40-4.19x. Its metric is the POOLED RMS
    over every element of every trajectory.

A mean of RMS values is not an RMS: pooling squares first, so it is pulled up
by whichever trajectories and epochs are worst, and the deployed error is
concentrated exactly there -- near the singularity, where the clock error
matters. Neither number is wrong. Reporting both without the bridge is.

WHAT THIS DOES
--------------
Evaluation only. Nothing is retrained and nothing is refitted: the same frozen
arms and the same converged clock head as `sc_decomposition.py`, on the same
frozen test split, scored under four aggregations at once --

  mean_of_slot_rms   mean over (run, tau) of the per-slot RMSE   [Table 11]
  pooled_rms         RMS over every element                      [Table 13]
  traj_median        median over runs of the per-trajectory RMSE
  traj_p90           90th percentile of the same

-- for the oracle arm (E_A), the deployed arm (E_dep) and the displacement (F).

Seeds are combined under the convention each aggregation deserves: pooled RMS
combines as an RMS over seeds, the means and the order statistics as means, so
no aggregation is silently converted into another on the way out.

Output: results/statecond/metric_bridge.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from sc_experiment import Norm, tensors, load_checkpoint
import statecond.data as scd

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(RES, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']


def fields_at(net, cond_fn, u0, te, n_tau, dev):
    out = []
    ri = np.asarray(te)
    for j in range(n_tau):
        ti = np.full(len(te), j)
        with torch.no_grad():
            out.append(net(u0[ri], cond_fn(ri, ti)).cpu().numpy())
    return np.stack(out, axis=1)          # (runs, tau, channel, x)


def aggregations(err):
    """Every aggregation this paper uses, on one error array.

    `err` is (runs, tau, channel, x). The per-slot RMSE reduces (channel, x);
    the per-trajectory RMSE reduces (tau, channel, x).
    """
    slot = np.sqrt((err ** 2).mean(axis=(2, 3)))          # (runs, tau)
    traj = np.sqrt((err ** 2).mean(axis=(1, 2, 3)))       # (runs,)
    return {
        'mean_of_slot_rms': float(slot.mean()),
        'pooled_rms': float(np.sqrt((err ** 2).mean())),
        'traj_mean': float(traj.mean()),
        'traj_median': float(np.median(traj)),
        'traj_p90': float(np.percentile(traj, 90)),
    }


def run(name, backbone, seeds, dev):
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    u, u0, par, phi, flag, lts = tensors(d, n, dev)
    taus = scd.tau_grid()
    n_tau = len(taus)
    taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)
    G = u[np.asarray(te)].cpu().numpy()

    rows = []
    for s in range(seeds):
        tau_ck = os.path.join(CKPT, f'{name}_{backbone}_tau_seed{s}.pt')
        head_ck = os.path.join(CKPT, f'{name}_fno_time_conv_seed{s}.pt')
        if not (os.path.exists(tau_ck) and os.path.exists(head_ck)):
            continue
        net, _ = load_checkpoint(tau_ck, dev)
        net.eval()
        head, _ = load_checkpoint(head_ck, dev)
        head.eval()

        with torch.no_grad():
            lts_hat = head(u0, par).squeeze(-1)
        # x n.t_sd: both terms are normalised log-time (see sc_decomposition).
        ratio = torch.pow(10.0, (lts - lts_hat) * n.t_sd).cpu().numpy()
        ri = np.asarray(te)
        tq = np.clip(taus[None, :] * ratio[ri][:, None], 0.0, 1.0)

        def cond_true(r, t):
            return torch.cat([par[r], taus_t[t][:, None]], 1)

        pos = {int(x): k for k, x in enumerate(ri)}

        def cond_dep(r, t):
            q = torch.tensor(tq[[pos[int(x)] for x in r], t],
                             dtype=torch.float32, device=dev)
            return torch.cat([par[r], q[:, None]], 1)

        Gv = np.stack([np.stack([np.interp(tq[i], taus,
                                           u[ri[i], :, c, x].cpu().numpy())
                                 for x in range(u.shape[3])], -1)
                       for i in range(len(ri)) for c in range(u.shape[2])])
        Gv = Gv.reshape(len(ri), u.shape[2], n_tau, u.shape[3]).transpose(0, 2, 1, 3)

        Ghat_v = fields_at(net, cond_true, u0, te, n_tau, dev)
        Ghat_dep = fields_at(net, cond_dep, u0, te, n_tau, dev)

        rows.append({
            'seed': s,
            'E_A': aggregations(Ghat_v - G),
            'E_dep': aggregations(Ghat_dep - G),
            'F': aggregations(Gv - G),
            'n_test': int(len(ri)), 'n_tau': int(n_tau),
            'n_elements': int(G.size),
        })
        del net, head
        torch.cuda.empty_cache()
    return rows


def combine(rows):
    """Seeds combined under each aggregation's own convention.

    `pooled_rms` is quadratic, so it combines as an RMS over seeds -- the same
    rule `sc_decomposition.py` uses, and the reason the identity closes there.
    The means and order statistics combine as means. Converting one into the
    other on the way out is exactly the mistake this script exists to expose.
    """
    keys = ['mean_of_slot_rms', 'pooled_rms', 'traj_mean', 'traj_median',
            'traj_p90']
    out = {}
    for term in ('E_A', 'E_dep', 'F'):
        out[term] = {}
        for k in keys:
            v = [r[term][k] for r in rows]
            out[term][k] = (float(np.sqrt(np.mean(np.square(v))))
                            if k == 'pooled_rms' else float(np.mean(v)))
    out['penalty'] = {k: out['E_dep'][k] / out['E_A'][k] for k in keys}
    out['F_over_E_A'] = {k: out['F'][k] / out['E_A'][k] for k in keys}
    out['n_test'] = rows[0]['n_test']
    out['n_elements'] = rows[0]['n_elements']
    out['seeds'] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=SYSTEMS)
    ap.add_argument('--backbones', nargs='+',
                    default=['fno', 'cno', 'transformer'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    torch.set_num_threads(2)              # F9 owns the CPU
    out = {'_metrics': {
        'mean_of_slot_rms': 'mean over (trajectory, tau) of the per-slot RMSE '
                            'over (channel, radius); the convention of '
                            'timehead.json and Table tab:clock',
        'pooled_rms': 'RMS over every element of every trajectory; the '
                      'convention of decomposition.json and Table '
                      'tab:decomposition',
        'traj_mean': 'mean over trajectories of the per-trajectory RMSE',
        'traj_median': 'median of the same',
        'traj_p90': '90th percentile of the same',
        'units': 'normalised field units (the Norm of the training split), '
                 'both channels weighted equally, identical for every row',
        'estimand': 'one frozen tau-arm operator, evaluated at the oracle tau '
                    '(E_A) and at tau_hat from the converged clock head '
                    '(E_dep); no arm is retrained for either column',
    }}
    for name in a.systems:
        for bb in a.backbones:
            rows = run(name, bb, a.seeds, a.device)
            if not rows:
                continue
            agg = combine(rows)
            out[f'{name}_{bb}'] = dict(agg, per_seed=rows)
            p = agg['penalty']
            print(f"{name:8s} {bb:12s} penalty  slot-mean {p['mean_of_slot_rms']:.3f}"
                  f"  pooled {p['pooled_rms']:.3f}"
                  f"  traj-median {p['traj_median']:.3f}"
                  f"  traj-p90 {p['traj_p90']:.3f}", flush=True)

    pts = [k for k in out if not k.startswith('_')]
    summ = {}
    for k in ('mean_of_slot_rms', 'pooled_rms', 'traj_median', 'traj_p90'):
        v = [out[p]['penalty'][k] for p in pts]
        summ[k] = {'min': float(min(v)), 'max': float(max(v)),
                   'median': float(np.median(v)),
                   'rms_log10': float(np.sqrt(np.mean(np.log10(v) ** 2)))}
    out['_summary'] = {
        'n_points': len(pts), 'penalty_range': summ,
        'reading': 'rms_log10 is the score the no-penalty predictor R_unit '
                   'achieves under that aggregation: predicting no penalty '
                   'means predicting log10(E_dep/E_A) = 0, so its RMS log '
                   'residual IS the RMS of the log penalties.',
    }
    json.dump(out, open(os.path.join(RES, 'metric_bridge.json'), 'w'), indent=1)
    print('\nrange of the deployment penalty over '
          f'{len(pts)} system-backbone points:')
    for k, v in summ.items():
        print(f"  {k:18s} {v['min']:.2f}-{v['max']:.2f}x   "
              f"median {v['median']:.2f}x   RMS log {v['rms_log10']:.4f} dex")
    print('wrote metric_bridge.json')


if __name__ == '__main__':
    main()
