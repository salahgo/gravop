"""E28 — a phase-conditioned production ensemble, saved for downstream use.

E20 showed that conditioning the SIDM emulator on an observable state
coordinate instead of on tau removes the deployment penalty, but it trained
throwaway models and kept only the metrics. E9 (inference) and E11 (N-body)
therefore still run on the *deployed tau-conditioned* ensemble in
`results/runs/gravop_s{0..4}` — now known to be the weakest arm. The paper
would otherwise argue for state-conditioning while its headline application
results were obtained without it, which is the largest coherence gap in the
project (P2).

This script trains the same five-member ensemble under phase-conditioning and
writes it in the checkpoint schema `scripts/train.py` uses, so `load_model`
picks it up unchanged and E9/E11 can switch arms with a flag rather than a
rewrite.

What is stored beyond the tau-conditioned schema:
  args['cond_dim'] = 5   (par, model_id, phi, flag) rather than 4
  norm['p_mu'], norm['p_sd']   phase normalisation, fitted on the training
                               split like every other constant here

The conditioning variable is E20's, unchanged and re-imported rather than
re-derived: phi = log10(rho_c(t)/rho_c(0)) with a post-minimum flag, both
readable off a density profile.

Usage:  e28_phase_ensemble.py [--seeds 5] [--epochs 200] [--device cuda:0]
Output: results/runs/gravop_phase_s{0..4}/{model.pt,metrics.json}
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import dataset as ds                      # noqa: E402
from gravop.model import GravOpNet                    # noqa: E402
from train import load_data, split, Norm              # noqa: E402
from e20_phase_conditioned import phase_variables     # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
RUNS = os.path.join(ROOT, 'results', 'runs')


def train_one(d, tr, te, n, seed, dev, epochs, phi_index=0):
    """One phase-conditioned member. Mirrors e20_phase_conditioned.run('phase')
    exactly, so the ensemble is the thing E20 measured rather than a variant."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_tau = d['lrho'].shape[1]

    lrho = torch.tensor((d['lrho'] - n.rho_mu) / n.rho_sd).float().to(dev)
    lv = torch.tensor((d['lv'] - n.v_mu) / n.v_sd).float().to(dev)
    prof = torch.stack([
        torch.tensor((d['lrho0'] - n.rho_mu) / n.rho_sd),
        torch.tensor((d['lv0'] - n.v_mu) / n.v_sd)], 1).float().to(dev)
    par = torch.tensor(np.stack([
        (d['log_sigma0'] - n.sig_mu) / n.sig_sd,
        (d['log_w'] - n.w_mu) / n.w_sd,
        d['model_id']], 1)).float().to(dev)

    p_np, flag_np = phase_variables(d['lrho'], r_index=phi_index)
    p_mu = float(p_np[tr].mean())
    p_sd = float(p_np[tr].std()) + 1e-9
    p_t = torch.tensor((p_np - p_mu) / p_sd).float().to(dev)
    flag_t = torch.tensor(flag_np).float().to(dev)

    net = GravOpNet(n_r=ds.N_R, cond_dim=5).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def cond_for(ri, ti):
        return torch.cat([par[ri], p_t[ri, ti][:, None],
                          flag_t[ri, ti][:, None]], 1)

    pr_ = np.repeat(tr, n_tau)
    pt_ = np.tile(np.arange(n_tau), len(tr))
    t0 = time.time()
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(len(pr_))
        for s in range(0, len(perm), 1024):
            sel = perm[s:s + 1024]
            ri, ti = pr_[sel], pt_[sel]
            f, _ = net(prof[ri], cond_for(ri, ti))
            tgt = torch.stack([lrho[ri, ti], lv[ri, ti]], 1)
            opt.zero_grad()
            nn.functional.mse_loss(f, tgt).backward()
            opt.step()
        sch.step()
    wall = time.time() - t0

    net.eval()
    errs = np.zeros((len(te), n_tau))
    with torch.no_grad():
        for j in range(n_tau):
            ri, ti = np.asarray(te), np.full(len(te), j)
            f, _ = net(prof[ri], cond_for(ri, ti))
            pr = f[:, 0].cpu().numpy() * n.rho_sd + n.rho_mu
            errs[:, j] = np.abs(10 ** (pr - d['lrho'][te, j]) - 1).mean(1)
    late = ds.tau_grid() > 0.92
    metrics = {'mean_pct': float(100 * errs.mean()),
               'median_pct': float(100 * np.median(errs.mean(1))),
               'late_mean_pct': float(100 * errs[:, late].mean()),
               'fit_seconds': wall}
    return net, {'p_mu': p_mu, 'p_sd': p_sd}, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--phi-index', type=int, default=0,
                    help='radial grid index the phase is read at. 0 is E20 and '
                         'the cross-domain protocol; 23 (r = 0.02 r_s) is the '
                         'innermost radius E9 observes, so an E9 arm '
                         'conditioned on index 0 would extrapolate')
    ap.add_argument('--suffix', default='',
                    help='appended to the run directory name, to keep '
                         'ensembles with different phi definitions apart')
    args = ap.parse_args()
    dev = torch.device(args.device)

    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['log_tc']))
    n = Norm(d, tr)

    summary = []
    for s in range(args.seeds):
        net, pnorm, metrics = train_one(d, tr, te, n, s, dev, args.epochs,
                                        args.phi_index)
        out_dir = os.path.join(RUNS, f'gravop_phase{args.suffix}_s{s}')
        os.makedirs(out_dir, exist_ok=True)
        norm = {**n.to_dict(), **pnorm}
        # 'model': 'gravop' so load_model builds the same backbone; cond_dim is
        # what tells it this member reads a phase rather than a tau
        ck_args = {'model': 'gravop', 'width': 64, 'modes': 32, 'layers': 4,
                   'cond_dim': 5, 'cond_kind': 'phase', 'seed': s,
                   'epochs': args.epochs, 'phi_index': args.phi_index}
        torch.save({'model': net.state_dict(), 'norm': norm,
                    'args': ck_args, 'split_seed': 1234},
                   os.path.join(out_dir, 'model.pt'))
        json.dump({'args': ck_args, 'norm': norm, 'test': metrics,
                   'n_params': sum(p.numel() for p in net.parameters())},
                  open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=1)
        summary.append(metrics)
        print(f"seed {s}: mean {metrics['mean_pct']:.3f}%  "
              f"late {metrics['late_mean_pct']:.3f}%  "
              f"({metrics['fit_seconds']:.0f}s) -> {out_dir}", flush=True)

    print(f"\nensemble of {len(summary)}: mean "
          f"{np.mean([m['mean_pct'] for m in summary]):.3f}% "
          f"+-{np.std([m['mean_pct'] for m in summary]):.3f}")


if __name__ == '__main__':
    main()
