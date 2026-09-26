"""Train GravOp emulator or baselines on dataset D1.

Usage:
  train.py --data data/d1/train.h5 --model gravop --seed 0 --out results/runs/NAME
  --model: gravop (FNO + physics losses) | fno_nophys | deeponet | pcamlp
  --lam_hyd/--lam_mass/--lam_en: physics loss weights (0 disables)
  --raw_time: use tau = t/t_end instead of t/t_c normalization (ablation)

Split: by run, seeded permutation 80/10/10.
All metrics land in <out>/metrics.json; checkpoint in <out>/model.pt.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gravop.model import GravOpNet, DeepONet, phys_residuals  # noqa: E402
from gravop import dataset as ds  # noqa: E402


def load_data(path):
    with h5py.File(path, 'r') as h5:
        d = {k: h5[k][:] for k in ('lrho', 'lv', 'lrho0', 'lv0', 'log_sigma0',
                                   'log_w', 'model_id', 'log_tc',
                                   'no_collapse', 'status')}
    ok = d['status'] >= 0
    for k in d:
        d[k] = d[k][ok]
    return d


def split(n, seed=1234):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


class Norm:
    """Normalization constants from training set."""
    def __init__(self, d, tr):
        self.rho_mu = float(d['lrho'][tr].mean())
        self.rho_sd = float(d['lrho'][tr].std())
        self.v_mu = float(d['lv'][tr].mean())
        self.v_sd = float(d['lv'][tr].std())
        self.sig_mu = float(d['log_sigma0'][tr].mean())
        self.sig_sd = float(d['log_sigma0'][tr].std()) or 1.0
        self.w_mu = float(d['log_w'][tr].mean())
        self.w_sd = float(d['log_w'][tr].std()) or 1.0
        self.tc_mu = float(d['log_tc'][tr].mean())
        self.tc_sd = float(d['log_tc'][tr].std()) or 1.0

    def to_dict(self):
        return dict(self.__dict__)


def make_tensors(d, norm, device):
    """Returns per-run tensors on device."""
    lrho = (torch.tensor(d['lrho']) - norm.rho_mu) / norm.rho_sd
    lv = (torch.tensor(d['lv']) - norm.v_mu) / norm.v_sd
    prof = torch.stack([
        (torch.tensor(d['lrho0']) - norm.rho_mu) / norm.rho_sd,
        (torch.tensor(d['lv0']) - norm.v_mu) / norm.v_sd], dim=1)
    par = torch.stack([
        (torch.tensor(d['log_sigma0']) - norm.sig_mu) / norm.sig_sd,
        (torch.tensor(d['log_w']) - norm.w_mu) / norm.w_sd,
        torch.tensor(d['model_id'], dtype=torch.float64)], dim=1)
    tc = (torch.tensor(d['log_tc']) - norm.tc_mu) / norm.tc_sd
    nc = torch.tensor(d['no_collapse'])
    return [t.float().to(device) for t in (lrho, lv, prof, par, tc, nc)]


def batch_inputs(prof, par, taus, run_idx, tau_idx):
    """Assemble model inputs for (run, tau) pairs."""
    cond = torch.cat([par[run_idx],
                      taus[tau_idx][:, None]], dim=1)
    return prof[run_idx], cond


def evaluate(model, tensors, taus, idx, norm, device, bs=8192):
    """Per-run mean metrics over all taus."""
    lrho, lv, prof, par, tc, nc = tensors
    n_tau = taus.shape[0]
    pairs_r = np.repeat(idx, n_tau)
    pairs_t = np.tile(np.arange(n_tau), len(idx))
    se_rho, se_v, rel_rho = [], [], []
    preds_tc, preds_nc = {}, {}
    model.eval()
    with torch.no_grad():
        for s in range(0, len(pairs_r), bs):
            ri = pairs_r[s:s + bs]
            ti = pairs_t[s:s + bs]
            pi, ci = batch_inputs(prof, par, taus, ri, ti)
            fields, scalars = model(pi, ci)
            tgt = torch.stack([lrho[ri, ti], lv[ri, ti]], dim=1)
            err = fields - tgt
            se_rho.append(err[:, 0].pow(2).mean(dim=1).cpu())
            se_v.append(err[:, 1].pow(2).mean(dim=1).cpu())
            # relative error on rho in linear space
            pr = fields[:, 0] * norm.rho_sd + norm.rho_mu
            tr = tgt[:, 0] * norm.rho_sd + norm.rho_mu
            rel = (10 ** pr / 10 ** tr - 1).abs().mean(dim=1).cpu()
            rel_rho.append(rel)
            for j, r in enumerate(ri):
                if r not in preds_tc:
                    preds_tc[r] = scalars[j, 0].item()
                    preds_nc[r] = scalars[j, 1].item()
    se_rho = torch.cat(se_rho)
    se_v = torch.cat(se_v)
    rel_rho = torch.cat(rel_rho)
    # tc error (collapsing runs only)
    tc_true, tc_pred, nc_true, nc_pred = [], [], [], []
    for r in idx:
        nc_true.append(nc[r].item())
        nc_pred.append(preds_nc[r] > 0)
        if nc[r].item() == 0:
            tc_true.append(tc[r].item() * norm.tc_sd + norm.tc_mu)
            tc_pred.append(preds_tc[r] * norm.tc_sd + norm.tc_mu)
    tc_true, tc_pred = np.array(tc_true), np.array(tc_pred)
    tc_rel = np.abs(10 ** tc_pred / 10 ** tc_true - 1)
    # regime split: runs that reach a terminal state (collapse or Knudsen
    # floor) are the hard cases; runs cut at t_end are the slow-evolution ones
    nc_arr = np.array([nc[r].item() for r in idx], dtype=bool)
    per_run_rel = rel_rho.reshape(len(idx), -1).mean(dim=1).numpy() \
        if rel_rho.numel() == len(idx) * n_tau else None
    return {
        'rmse_lrho': float(se_rho.mean().sqrt()),
        'rmse_lv': float(se_v.mean().sqrt()),
        'mean_rel_rho': float(rel_rho.mean()),
        'median_rel_rho': float(rel_rho.median()),
        'median_rel_rho_terminal': (
            float(np.median(per_run_rel[~nc_arr]))
            if per_run_rel is not None and (~nc_arr).any() else None),
        'median_rel_rho_tend': (
            float(np.median(per_run_rel[nc_arr]))
            if per_run_rel is not None and nc_arr.any() else None),
        'tc_rel_err_mean': float(tc_rel.mean()) if len(tc_rel) else None,
        'tc_rel_err_median': float(np.median(tc_rel)) if len(tc_rel) else None,
        'nc_acc': float(np.mean(np.array(nc_true) == np.array(nc_pred))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/d1/train.h5')
    ap.add_argument('--model', default='gravop')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam_hyd', type=float, default=0.05)
    ap.add_argument('--lam_mass', type=float, default=0.1)
    ap.add_argument('--lam_en', type=float, default=0.02)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n_tau_train', type=int, default=0,
                    help='sparse temporal supervision: use only this many of '
                         'the tau samples per trajectory during training '
                         '(0 = all). Evaluation always uses the full grid.')
    ap.add_argument('--n_train', type=int, default=0,
                    help='subsample the training split to this many runs '
                         '(0 = use all); val/test splits are unchanged')
    args = ap.parse_args()

    if args.model in ('fno_nophys', 'deeponet', 'pcamlp'):
        args.lam_hyd = args.lam_mass = args.lam_en = 0.0

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    d = load_data(args.data)
    n = len(d['log_tc'])
    tr, va, te = split(n)
    if args.n_train and args.n_train < len(tr):
        # deterministic prefix of the (already permuted) training split, so
        # smaller budgets are nested subsets of larger ones
        tr = tr[:args.n_train]
    norm = Norm(d, tr)
    tensors = make_tensors(d, norm, device)
    lrho, lv, prof, par, tc, nc = tensors
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=device)
    log_r = torch.tensor(np.linspace(ds.LOGR_MIN, ds.LOGR_MAX, ds.N_R),
                         dtype=torch.float32, device=device)

    if args.model in ('gravop', 'fno_nophys'):
        model = GravOpNet(n_r=ds.N_R, width=args.width, modes=args.modes,
                          n_layers=args.layers).to(device)
    elif args.model == 'deeponet':
        model = DeepONet(n_r=ds.N_R).to(device)
    else:
        raise SystemExit('pcamlp handled in baselines_pcamlp.py')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_tau = taus.shape[0]
    # sparse temporal supervision: keep an evenly spaced subset of tau samples
    # (endpoints included) so the model still sees the collapse limit
    if args.n_tau_train and args.n_tau_train < n_tau:
        tau_idx = np.unique(np.linspace(0, n_tau - 1, args.n_tau_train)
                            .round().astype(int))
    else:
        tau_idx = np.arange(n_tau)
    pairs_r = np.repeat(tr, len(tau_idx))
    pairs_t = np.tile(tau_idx, len(tr))
    n_pairs = len(pairs_r)
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_pairs)
        tot, nb = 0.0, 0
        for s in range(0, n_pairs, args.bs):
            sel = perm[s:s + args.bs]
            ri, ti = pairs_r[sel], pairs_t[sel]
            pi, ci = batch_inputs(prof, par, taus, ri, ti)
            fields, scalars = model(pi, ci)
            tgt = torch.stack([lrho[ri, ti], lv[ri, ti]], dim=1)
            loss = F.mse_loss(fields, tgt)
            loss = loss + 0.1 * F.mse_loss(scalars[:, 0], tc[ri]) \
                + 0.1 * F.binary_cross_entropy_with_logits(
                    scalars[:, 1], nc[ri].float())
            if args.lam_hyd or args.lam_mass or args.lam_en:
                ramp = min(1.0, (ep + 1) / max(1, int(0.2 * args.epochs)))
                plrho = fields[:, 0] * norm.rho_sd + norm.rho_mu
                plv = fields[:, 1] * norm.v_sd + norm.v_mu
                hyd, mtot, etot = phys_residuals(plrho, plv, log_r)
                # reference conserved quantities from the *initial* profile
                ilrho = pi[:, 0] * norm.rho_sd + norm.rho_mu
                ilv = pi[:, 1] * norm.v_sd + norm.v_mu
                _, m0, e0 = phys_residuals(ilrho, ilv, log_r)
                dm = ((mtot - m0) / m0).abs().clamp(max=10.0)
                de = ((etot - e0) / m0.abs()).abs().clamp(max=10.0)
                loss = loss + ramp * (
                    args.lam_hyd * hyd.clamp(-10, 10).pow(2).mean()
                    + args.lam_mass * dm.pow(2).mean()
                    + args.lam_en * de.pow(2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            vm = evaluate(model, tensors, taus, va, norm, device)
            hist.append({'epoch': ep + 1, 'train_loss': tot / nb, **vm})
            print(f'ep {ep + 1} loss {tot / nb:.5f} '
                  f'val rmse_lrho {vm["rmse_lrho"]:.5f} '
                  f'rel_rho {vm["mean_rel_rho"]:.4f} '
                  f'tc {vm["tc_rel_err_median"]}', flush=True)

    test_m = evaluate(model, tensors, taus, te, norm, device)
    wall = time.time() - t0
    out = {'args': vars(args), 'norm': norm.to_dict(), 'wall_seconds': wall,
           'n_runs': int(n), 'split': [len(tr), len(va), len(te)],
           'history': hist, 'test': test_m,
           'n_params': sum(p.numel() for p in model.parameters())}
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=1)
    torch.save({'model': model.state_dict(), 'norm': norm.to_dict(),
                'args': vars(args), 'split_seed': 1234},
               os.path.join(args.out, 'model.pt'))
    print('TEST', json.dumps(test_m, indent=1))
    print(f'wall {wall:.0f}s')


if __name__ == '__main__':
    main()
