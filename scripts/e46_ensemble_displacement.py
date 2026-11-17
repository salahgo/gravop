#!/usr/bin/env python3
"""E46 -- a deployable per-trajectory displacement from clock-head seed disagreement.

WHY
---
E45 showed that the zero-coefficient rule p = sqrt(E_A^2 + F^2)/E_A scores 0.055 dex
(in sample) and 0.051 dex (reserved draws) only when F uses each scored trajectory's
REALISED clock error; with the validation error distribution it loosens to 0.088-0.155
dex, because clock error and trajectory sensitivity are correlated across trajectories.
The realised error needs the true terminal time. This asks whether a quantity available at
deployment -- the disagreement of the deployed clock head with the other two converged
seeds -- carries enough per-trajectory information to recover the realised-error score.

PROTOCOL (fixed before scoring; nothing is trained)
---------------------------------------------------
For system S, deployed head seed s, and every trajectory i (validation, scored):
  spread_i = |log10 t_hat_s(i) - mean_{s' != s} log10 t_hat_s'(i)|        (deployable)
  k_s      = median_val |log10 t_true - log10 t_hat_s| / median_val spread  (one scalar,
             validation split only; never the scored population)
  m_i      = 10**(k_s * spread_i) - 1, displacement taken with either sign:
             true/predicted ratios 1/(1+m_i) and 1/(1-m_i), as E45's `val_median` does
             with a single size.
F_ens is the RMS displacement over scored trajectories, both signs, seeds aggregated as
stored (pooled RMS per seed, RMS over seeds). The frozen rule is scored with the stored
E_A and E_dep points and aggregation (mean over systems of within-system RMS log10
residual), in sample (test split) and on the reserved draws. Compared, unchanged, with
E45's `realised`, `val_errors`, `val_median` and the fitted constant. Whatever the result,
it is reported.

Reads the same inputs as sc_deployable_displacement.py.
Output: results/statecond/ensemble_displacement.json
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.dirname(__file__))
import sc_deployable_displacement as e45  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUT = os.path.join(e45.RES, 'ensemble_displacement.json')


def log_heads(name, u0, par):
    out = []
    for s in range(e45.SEEDS):
        h, _ = e45.load_checkpoint(os.path.join(e45.CKPT, f'{name}_fno_time_conv_seed{s}.pt'), 'cpu'); h.eval()
        with torch.no_grad(): out.append(h(u0, par).squeeze(-1).numpy())
    return np.array(out)


def displacement(name, scored):
    d = e45.scd.load(os.path.join(e45.DATA, f'{name}.h5')); tr, va, te = e45.scd.split(d['u'].shape[0]); n = e45.Norm(d, tr)
    u, u0, par, _, _, lts = e45.tensors(d, n, 'cpu'); U = u.numpy(); va = np.asarray(va)
    H = log_heads(name, u0, par); L = lts.numpy()
    if scored == 'test':
        Us, Hs, Ls = U[te], H[:, te], L[te]
    else:
        dr = e45.scd.load(os.path.join(e45.DATA, 'reserved', f'{name}.h5')); ur, u0r, parr, _, _, ltr = e45.tensors(dr, n, 'cpu')
        Us, Hs, Ls = ur.numpy(), log_heads(name, u0r, parr), ltr.numpy()
    taus = e45.scd.tau_grid(); rows = []
    for s in range(e45.SEEDS):
        oth = [k for k in range(e45.SEEDS) if k != s]
        spr_v = np.abs(H[s, va] - H[oth][:, va].mean(0)) * n.t_sd; err_v = np.abs(H[s, va] - L[va]) * n.t_sd
        k = float(np.median(err_v) / np.median(spr_v))
        spr = np.abs(Hs[s] - Hs[oth].mean(0)) * n.t_sd; err = np.abs(Hs[s] - Ls) * n.t_sd
        m = np.minimum(10 ** (k * spr) - 1, .9)
        ratios = np.c_[1 / (1 + m), 1 / (1 - m)]
        rows.append(dict(seed=s, k=k, F_ens=float(np.sqrt(e45.mean_sq_displacement(Us, ratios, taus))),
                         spearman_val=float(spearmanr(err_v, spr_v)[0]), spearman_scored=float(spearmanr(err, spr)[0]),
                         median_err_scored=float(np.median(err)), median_pred_scored=float(np.median(k * spr))))
    return dict(F_ens=float(np.sqrt(np.mean([r['F_ens'] ** 2 for r in rows]))), per_seed=rows)


def main():
    torch.set_num_threads(2)
    dec = json.load(open(os.path.join(e45.RES, 'decomposition.json')))
    pro = json.load(open(os.path.join(ROOT, 'results', 'e43', 'prospective.json')))
    stored = json.load(open(os.path.join(e45.RES, 'deployable_displacement.json')))
    out = {'protocol': __doc__.split('Reads the same')[0].strip()}
    for regime, scored, points in (('in_sample', 'test', dec), ('reserved', 'reserved', pro['points'])):
        pts = {k: v for k, v in points.items() if k.rsplit('_', 1)[1] in e45.BACKBONES}
        F = {s: displacement(s, scored) for s in e45.SYSTEMS}
        sc = e45.score(pts, {s: F[s]['F_ens'] for s in e45.SYSTEMS})
        prev = stored[regime]; const = prev['nulls']['constant']
        out[regime] = dict(F=F, ensemble=sc,
                           compare={k: prev[k] for k in ('realised', 'val_errors', 'val_only', 'val_median')}, constant=const,
                           factor_vs_constant=const['mean_of_system_rms'] / sc['mean_of_system_rms'],
                           systems_behind_constant=sorted(s for s in e45.SYSTEMS if sc['per_system'][s] > const['per_system'][s]))
        print(regime, 'ensemble %.4f' % sc['mean_of_system_rms'], ' '.join(f"{k} {prev[k]['mean_of_system_rms']:.4f}" for k in ('realised', 'val_errors', 'val_median')),
              'constant %.4f' % const['mean_of_system_rms'], sc['per_system'])
    json.dump(out, open(OUT, 'w'), indent=1); print('wrote', os.path.relpath(OUT, ROOT))


if __name__ == '__main__': main()
