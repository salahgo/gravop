"""Classic-emulator baseline: PCA (POD) on trajectories + MLP on coefficients.

Trajectory vector per run: (lrho, lv) on the (tau x r) grid, flattened
(2*48*128 = 12288 dims) -> PCA to n_pod coefficients (fit on train split).
Initial profile encoded by its own PCA (n_ic comps). MLP maps
[ic coeffs, log sigma0, log w, model_id] -> [pod coeffs, log_tc, nc logit].
Metrics computed identically to train.py's evaluate (same split seed).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from train import load_data, split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/d1/train.h5')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_pod', type=int, default=128)
    ap.add_argument('--n_ic', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_data(args.data)
    n = len(d['log_tc'])
    tr, va, te = split(n)

    traj = np.concatenate([d['lrho'].reshape(n, -1),
                           d['lv'].reshape(n, -1)], axis=1).astype(np.float64)
    ic = np.concatenate([d['lrho0'], d['lv0']], axis=1).astype(np.float64)

    mu_t = traj[tr].mean(0)
    A = traj[tr] - mu_t
    # PCA via SVD on train rows
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[:args.n_pod]
    evr = float((S[:args.n_pod] ** 2).sum() / (S ** 2).sum())
    mu_i = ic[tr].mean(0)
    Ui, Si, Vti = np.linalg.svd(ic[tr] - mu_i, full_matrices=False)
    Vi = Vti[:args.n_ic]

    def enc_t(x):
        return (x - mu_t) @ V.T

    def enc_i(x):
        return (x - mu_i) @ Vi.T

    par = np.stack([d['log_sigma0'], d['log_w'], d['model_id']], axis=1)
    X = np.concatenate([enc_i(ic), par], axis=1)
    Y = enc_t(traj)
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-9
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-9
    tcm, tcs = d['log_tc'][tr].mean(), d['log_tc'][tr].std()

    dev = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32, device=dev)
    Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32, device=dev)
    tct = torch.tensor((d['log_tc'] - tcm) / tcs, dtype=torch.float32,
                       device=dev)
    nct = torch.tensor(d['no_collapse'], dtype=torch.float32, device=dev)

    mlp = nn.Sequential(
        nn.Linear(X.shape[1], 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, args.n_pod + 2)).to(dev)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    tr_t = torch.tensor(tr, device=dev)
    t0 = time.time()
    for ep in range(args.epochs):
        mlp.train()
        perm = tr_t[torch.randperm(len(tr), device=dev)]
        for s in range(0, len(perm), 512):
            sel = perm[s:s + 512]
            out = mlp(Xt[sel])
            loss = F.mse_loss(out[:, :args.n_pod], Yt[sel]) \
                + 0.1 * F.mse_loss(out[:, args.n_pod], tct[sel]) \
                + 0.1 * F.binary_cross_entropy_with_logits(
                    out[:, args.n_pod + 1], nct[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    wall = time.time() - t0

    # evaluate on test split with the same metric definitions as train.py
    mlp.eval()
    with torch.no_grad():
        out = mlp(Xt[te]).cpu().numpy()
    Yp = out[:, :args.n_pod] * ys + ym
    traj_p = Yp @ V + mu_t
    n_tau, n_r = d['lrho'].shape[1], d['lrho'].shape[2]
    lrho_p = traj_p[:, :n_tau * n_r].reshape(-1, n_tau, n_r)
    lv_p = traj_p[:, n_tau * n_r:].reshape(-1, n_tau, n_r)
    lrho_t = d['lrho'][te]
    lv_t = d['lv'][te]
    # normalized-field RMSE consistent with train.py (using its norm stats)
    rho_mu, rho_sd = d['lrho'][tr].mean(), d['lrho'][tr].std()
    v_mu, v_sd = d['lv'][tr].mean(), d['lv'][tr].std()
    rmse_lrho = float(np.sqrt((((lrho_p - lrho_t) / rho_sd) ** 2).mean()))
    rmse_lv = float(np.sqrt((((lv_p - lv_t) / v_sd) ** 2).mean()))
    rel = np.abs(10 ** (lrho_p - lrho_t) - 1)
    tc_p = out[:, args.n_pod] * tcs + tcm
    coll = d['no_collapse'][te] == 0
    tc_rel = np.abs(10 ** (tc_p[coll] - d['log_tc'][te][coll]) - 1)
    nc_pred = out[:, args.n_pod + 1] > 0
    metrics = {
        'rmse_lrho': rmse_lrho, 'rmse_lv': rmse_lv,
        'mean_rel_rho': float(rel.mean()),
        'median_rel_rho': float(np.median(rel)),
        'tc_rel_err_mean': float(tc_rel.mean()),
        'tc_rel_err_median': float(np.median(tc_rel)),
        'nc_acc': float((nc_pred == (d['no_collapse'][te] == 1)).mean()),
        'pod_explained_var': evr,
    }
    outd = {'args': vars(args), 'test': metrics, 'wall_seconds': wall,
            'n_params': sum(p.numel() for p in mlp.parameters())}
    json.dump(outd, open(os.path.join(args.out, 'metrics.json'), 'w'),
              indent=1)
    print('TEST', json.dumps(metrics, indent=1))


if __name__ == '__main__':
    main()
