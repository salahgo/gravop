"""E20 — phase-conditioned emulation: remove the deployment penalty.

E17 established that the emulator's deployed error (6.98%) is 4.5x its
oracle-phase error (1.54%), because to answer "what does this halo look like at
physical time t?" one must first predict the termination time t_norm, and near
the singularity a small t_norm error queries a visibly wrong phase. E18 showed
that only ~1 of the 5.4 lost points is recoverable by predicting t_norm better:
the penalty is intrinsic to conditioning on *time*.

The proposal here is to stop conditioning on time. An observer looking at a halo
does not know tau, but does know the halo's *state*: in particular the central
density contrast relative to the initial profile. That is an observable phase
indicator. If the operator is conditioned on it instead of on tau, the
prediction never needs t_norm, and the deployment penalty disappears by
construction rather than by regression.

Conditioning variable
    p     = log10( rho_c(t) / rho_c(0) )     central density contrast
    flag  = 1 after the epoch of minimum central density, else 0
The flag is needed because the contrast is two-valued: the core first expands
(p decreasing) and then collapses (p increasing), so p alone does not identify
the phase. Both are measurable from a density profile.

This script trains the same FNO backbone under both conditionings on identical
splits and seeds, and compares them on the metric that matters: error when the
phase must be determined at prediction time, not handed to the model.

Usage: e20_phase_conditioned.py [--seeds 3] [--device cuda:1]
Output: results/e20/phase_conditioned.json
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

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e20')
os.makedirs(OUT, exist_ok=True)


def phase_variables(lrho, r_index=0):
    """Central-density contrast and post-minimum flag from the field itself.

    lrho: (n_runs, n_tau, n_r) log10 density. Uses grid point `r_index` as the
    central density -- by default the innermost, which is what an observer
    reads off a resolved profile.

    `r_index` is a parameter because "observable" depends on the observation.
    The emulator's grid starts at 0.0032 r_s, but E9's mock profile is sampled
    from 0.02 r_s outward, so a phase read at index 0 is NOT contained in E9's
    data and conditioning on it there would be an extrapolation rather than a
    measurement. Index 23 is the first grid point E9 actually observes.
    """
    lc = lrho[:, :, r_index]                 # log10 rho_c(tau)
    p = lc - lc[:, :1]                       # contrast relative to tau = 0
    imin = np.argmin(lc, axis=1)             # epoch of minimum central density
    tau_idx = np.arange(lrho.shape[1])[None, :]
    flag = (tau_idx > imin[:, None]).astype(np.float32)
    return p.astype(np.float32), flag


def run(cond_kind, d, tr, va, te, n, seed, device, epochs=200):
    """Train the same backbone with tau- or phase-conditioning."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
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

    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=dev)
    p_np, flag_np = phase_variables(d['lrho'])
    p_mu, p_sd = p_np[tr].mean(), p_np[tr].std() + 1e-9
    p_t = torch.tensor((p_np - p_mu) / p_sd).float().to(dev)
    flag_t = torch.tensor(flag_np).float().to(dev)

    # tau-conditioning uses 4 conditioning inputs; phase-conditioning uses 5
    cond_dim = 4 if cond_kind == 'tau' else 5
    net = GravOpNet(n_r=ds.N_R, cond_dim=cond_dim).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def cond_for(ri, ti):
        if cond_kind == 'tau':
            return torch.cat([par[ri], taus[ti][:, None]], 1)
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
            ri = np.asarray(te)
            ti = np.full(len(te), j)
            f, _ = net(prof[ri], cond_for(ri, ti))
            pr = f[:, 0].cpu().numpy() * n.rho_sd + n.rho_mu
            tgt = d['lrho'][te, j]
            errs[:, j] = np.abs(10 ** (pr - tgt) - 1).mean(1)
    late = ds.tau_grid() > 0.92
    return {'mean_pct': float(100 * errs.mean()),
            'median_pct': float(100 * np.median(errs.mean(1))),
            'late_mean_pct': float(100 * errs[:, late].mean()),
            'fit_seconds': wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--device', default='cuda:1')
    args = ap.parse_args()

    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['log_tc']))
    n = Norm(d, tr)

    res = {}
    for kind in ('tau', 'phase'):
        runs = [run(kind, d, tr, va, te, n, s, args.device)
                for s in range(args.seeds)]
        res[kind] = {k: float(np.mean([r[k] for r in runs]))
                     for k in runs[0]}
        res[kind]['std_mean_pct'] = float(np.std([r['mean_pct']
                                                  for r in runs]))
        print(f"{kind:6s}: mean {res[kind]['mean_pct']:.3f}% "
              f"(+-{res[kind]['std_mean_pct']:.3f})  "
              f"late {res[kind]['late_mean_pct']:.3f}%", flush=True)

    # The comparison that matters. tau-conditioning must predict t_norm at
    # deployment (E17: 6.975% mean, 13.094% late). Phase-conditioning reads its
    # conditioning variable off the observed profile, so its measured error IS
    # its deployed error.
    e17 = json.load(open(os.path.join(ROOT, 'results', 'e17',
                                      'error_budget.json')))
    res['tau_deployed_from_e17'] = {
        'mean_pct': e17['B_predicted_tnorm']['mean_pct'],
        'late_mean_pct': e17['B_predicted_tnorm']['late_mean_pct']}
    res['verdict'] = {
        'phase_vs_tau_deployed_mean': float(
            e17['B_predicted_tnorm']['mean_pct'] / res['phase']['mean_pct']),
        'phase_vs_tau_deployed_late': float(
            e17['B_predicted_tnorm']['late_mean_pct']
            / res['phase']['late_mean_pct'])}
    print(f"\nAt deployment (phase must be determined, not given):")
    print(f"  tau-conditioned  : {res['tau_deployed_from_e17']['mean_pct']:.3f}% "
          f"mean, {res['tau_deployed_from_e17']['late_mean_pct']:.3f}% late")
    print(f"  phase-conditioned: {res['phase']['mean_pct']:.3f}% mean, "
          f"{res['phase']['late_mean_pct']:.3f}% late")
    print(f"  improvement: {res['verdict']['phase_vs_tau_deployed_mean']:.2f}x "
          f"mean, {res['verdict']['phase_vs_tau_deployed_late']:.2f}x late")
    json.dump(res, open(os.path.join(OUT, 'phase_conditioned.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
