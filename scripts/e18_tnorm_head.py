"""E18 — can the t_norm bottleneck identified in E17 be reduced?

E17 showed that the deployed field error (6.98%) is 4.5x the oracle-t_norm
error (1.54%), because near the collapse singularity the profile changes so
fast that even a 0.34% error in the predicted termination time queries a
visibly wrong phase of the evolution.

Two questions:
  1. How much of that 5.4-point gap is recoverable by predicting t_norm better?
     We train a dedicated regressor (the production estimate comes from a head
     that shares a trunk with the field task, so it may be under-fitted) and
     propagate its error through the same decomposition.
  2. What is the floor? Even a perfect t_norm leaves the 1.54% oracle error.

Output: results/e18/tnorm_head.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import dataset as ds                      # noqa: E402
from train import load_data, split, make_tensors      # noqa: E402
from evaluate_all import load_model, NormObj          # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e18')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def build_features(d, idx, n):
    """Inputs available at prediction time: parameters + initial profile."""
    par = np.stack([(d['log_sigma0'][idx] - n.sig_mu) / n.sig_sd,
                    (d['log_w'][idx] - n.w_mu) / n.w_sd,
                    d['model_id'][idx]], axis=1)
    prof = np.concatenate([(d['lrho0'][idx] - n.rho_mu) / n.rho_sd,
                           (d['lv0'][idx] - n.v_mu) / n.v_sd], axis=1)
    return np.concatenate([par, prof], axis=1).astype(np.float32)


def train_regressor(X, y, Xv, yv, seed=0, epochs=3000, width=256):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], width), nn.GELU(),
                        nn.Linear(width, width), nn.GELU(),
                        nn.Linear(width, width), nn.GELU(),
                        nn.Linear(width, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X, device=DEV); yt = torch.tensor(y, device=DEV)[:, None]
    Xvt = torch.tensor(Xv, device=DEV)
    best, best_state = 1e9, None
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xt), yt)
        loss.backward(); opt.step(); sch.step()
        if (ep + 1) % 100 == 0:
            net.eval()
            with torch.no_grad():
                v = float(nn.functional.mse_loss(
                    net(Xvt), torch.tensor(yv, device=DEV)[:, None]))
            if v < best:
                best, best_state = v, {k: p.clone()
                                       for k, p in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net


def main():
    models, norm = [], None
    for s in range(5):
        m, nrm, _ = load_model(os.path.join(ROOT, 'results', 'runs',
                                            f'gravop_s{s}'))
        models.append(m); norm = nrm
    n = NormObj(norm)
    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['log_tc']))
    lrho, lv, prof, par, tc, nc = make_tensors(d, n, DEV)
    taus_np = ds.tau_grid()

    # --- production scalar head error on test
    with torch.no_grad():
        c0 = torch.cat([par[te], torch.zeros(len(te), 1, device=DEV)], 1)
        ltc_head = np.mean([m(prof[te], c0)[1][:, 0].cpu().numpy()
                            for m in models], axis=0) * n.tc_sd + n.tc_mu
    ltc_true = d['log_tc'][te]
    err_head = np.abs(10 ** (ltc_head - ltc_true) - 1)

    # --- dedicated regressor, 5-member ensemble
    ym, ysd = d['log_tc'][tr].mean(), d['log_tc'][tr].std()
    Xtr, Xva, Xte = (build_features(d, i, n) for i in (tr, va, te))
    ytr = ((d['log_tc'][tr] - ym) / ysd).astype(np.float32)
    yva = ((d['log_tc'][va] - ym) / ysd).astype(np.float32)
    preds = []
    for s in range(5):
        net = train_regressor(Xtr, ytr, Xva, yva, seed=s)
        with torch.no_grad():
            preds.append(net(torch.tensor(Xte, device=DEV))[:, 0].cpu().numpy())
    ltc_ded = np.mean(preds, axis=0) * ysd + ym
    err_ded = np.abs(10 ** (ltc_ded - ltc_true) - 1)

    print(f'median |t_norm error|:  production head {100*np.median(err_head):.3f}%'
          f'   dedicated regressor {100*np.median(err_ded):.3f}%')

    # --- propagate both through the field operator
    def field_err(ratio):
        tau_q = np.clip(np.tile(taus_np, (len(te), 1)) / ratio[:, None], 0, 1)
        errs = np.zeros((len(te), len(taus_np)))
        with torch.no_grad():
            for j in range(len(taus_np)):
                q = torch.tensor(tau_q[:, j], dtype=torch.float32,
                                 device=DEV)[:, None]
                cond = torch.cat([par[te], q], 1)
                pr = np.mean([m(prof[te], cond)[0][:, 0].cpu().numpy()
                              for m in models], axis=0)
                tgt = lrho[te, j].cpu().numpy()
                errs[:, j] = np.abs(10 ** ((pr - tgt) * n.rho_sd) - 1).mean(1)
        return errs

    late = taus_np > 0.92
    res = {}
    for tag, ratio in (('oracle', np.ones(len(te))),
                       ('production_head', 10 ** (ltc_head - ltc_true)),
                       ('dedicated_regressor', 10 ** (ltc_ded - ltc_true))):
        e = field_err(ratio)
        res[tag] = {'mean_pct': float(100 * e.mean()),
                    'late_mean_pct': float(100 * e[:, late].mean())}
        print(f'  {tag:22s} field error {res[tag]["mean_pct"]:6.3f}%'
              f'   (late {res[tag]["late_mean_pct"]:6.3f}%)')

    out = {'tnorm_median_err_pct': {
               'production_head': float(100 * np.median(err_head)),
               'dedicated_regressor': float(100 * np.median(err_ded))},
           'field_error': res,
           'recoverable_points': float(res['production_head']['mean_pct']
                                       - res['dedicated_regressor']['mean_pct']),
           'floor_points': float(res['oracle']['mean_pct'])}
    json.dump(out, open(os.path.join(OUT, 'tnorm_head.json'), 'w'), indent=1)
    print(f"\nrecovered {out['recoverable_points']:+.3f} points; "
          f"floor (perfect t_norm) is {out['floor_points']:.3f}%")


if __name__ == '__main__':
    main()
