"""E21 — does phase-conditioning survive a noisy phase measurement?

E20 showed that conditioning the operator on an observable phase indicator
(central density contrast plus a post-minimum flag) instead of on normalised
time is 10.9x better at deployment. But that result assumed the indicator is
known exactly. A real observer measures it from a noisy density profile, which
converts a prediction problem into a measurement-error problem. Until that is
quantified the 0.64% of E20 is an upper bound, not a deployment figure. This is
the limitation flagged in an earlier project review.

Protocol: train once with the exact indicator (as in E20), then evaluate with
the indicator perturbed by Gaussian noise of increasing amplitude, in dex,
spanning plausible observational precision on a central density. The
post-minimum flag is additionally corrupted at a rate reflecting how hard it is
to tell whether a halo is pre- or post-core-minimum from a single epoch.

The comparison point is the tau-conditioned model at deployment (6.98% mean,
13.09% late, from E17): phase-conditioning is only the better choice while its
noisy error stays below that.

Output: results/e21/phase_noise.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import dataset as ds                       # noqa: E402
from gravop.model import GravOpNet                     # noqa: E402
from train import load_data, split, Norm               # noqa: E402
from e20_phase_conditioned import phase_variables      # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e21')
os.makedirs(OUT, exist_ok=True)

# noise on the central-density contrast, in dex. 0.02 dex ~ 5%, 0.1 dex ~ 26%.
NOISE_DEX = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
FLAG_ERROR_RATES = [0.0, 0.05, 0.15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=200)
    args = ap.parse_args()
    dev = torch.device(args.device)

    d = load_data(os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    tr, va, te = split(len(d['log_tc']))
    n = Norm(d, tr)
    n_tau = d['lrho'].shape[1]

    lrho = torch.tensor((d['lrho'] - n.rho_mu) / n.rho_sd).float().to(dev)
    lv = torch.tensor((d['lv'] - n.v_mu) / n.v_sd).float().to(dev)
    prof = torch.stack([
        torch.tensor((d['lrho0'] - n.rho_mu) / n.rho_sd),
        torch.tensor((d['lv0'] - n.v_mu) / n.v_sd)], 1).float().to(dev)
    par = torch.tensor(np.stack([
        (d['log_sigma0'] - n.sig_mu) / n.sig_sd,
        (d['log_w'] - n.w_mu) / n.w_sd, d['model_id']], 1)).float().to(dev)

    p_np, flag_np = phase_variables(d['lrho'])
    p_mu, p_sd = p_np[tr].mean(), p_np[tr].std() + 1e-9

    results = {f'{nd}': {f'{fr}': [] for fr in FLAG_ERROR_RATES}
               for nd in NOISE_DEX}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = GravOpNet(n_r=ds.N_R, cond_dim=5).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        p_t = torch.tensor((p_np - p_mu) / p_sd).float().to(dev)
        flag_t = torch.tensor(flag_np).float().to(dev)
        pr_ = np.repeat(tr, n_tau)
        pt_ = np.tile(np.arange(n_tau), len(tr))
        for _ in range(args.epochs):
            net.train()
            perm = np.random.permutation(len(pr_))
            for s in range(0, len(perm), 1024):
                sel = perm[s:s + 1024]
                ri, ti = pr_[sel], pt_[sel]
                cond = torch.cat([par[ri], p_t[ri, ti][:, None],
                                  flag_t[ri, ti][:, None]], 1)
                f, _ = net(prof[ri], cond)
                tgt = torch.stack([lrho[ri, ti], lv[ri, ti]], 1)
                opt.zero_grad()
                nn.functional.mse_loss(f, tgt).backward()
                opt.step()
            sch.step()
        net.eval()

        rng = np.random.default_rng(1000 + seed)
        for nd in NOISE_DEX:
            for fr in FLAG_ERROR_RATES:
                errs = np.zeros((len(te), n_tau))
                with torch.no_grad():
                    for j in range(n_tau):
                        ri = np.asarray(te)
                        ti = np.full(len(te), j)
                        pn = p_np[ri, ti] + nd * rng.standard_normal(len(ri))
                        fl = flag_np[ri, ti].copy()
                        flip = rng.random(len(ri)) < fr
                        fl[flip] = 1.0 - fl[flip]
                        cond = torch.cat([
                            par[ri],
                            torch.tensor((pn - p_mu) / p_sd,
                                         dtype=torch.float32, device=dev)[:, None],
                            torch.tensor(fl, dtype=torch.float32,
                                         device=dev)[:, None]], 1)
                        f, _ = net(prof[ri], cond)
                        pr = f[:, 0].cpu().numpy() * n.rho_sd + n.rho_mu
                        errs[:, j] = np.abs(10 ** (pr - d['lrho'][te, j]) - 1
                                            ).mean(1)
                late = ds.tau_grid() > 0.92
                results[f'{nd}'][f'{fr}'].append(
                    {'mean_pct': float(100 * errs.mean()),
                     'late_mean_pct': float(100 * errs[:, late].mean())})
        print(f'seed {seed} done', flush=True)

    out = {'noise_dex': NOISE_DEX, 'flag_error_rates': FLAG_ERROR_RATES,
           'tau_deployed_reference': {'mean_pct': 6.975, 'late_mean_pct': 13.094},
           'grid': {}}
    print(f"\n{'noise[dex]':>11} " +
          " ".join(f'flag_err={f:<5.0%}' for f in FLAG_ERROR_RATES))
    for nd in NOISE_DEX:
        row = []
        out['grid'][f'{nd}'] = {}
        for fr in FLAG_ERROR_RATES:
            runs = results[f'{nd}'][f'{fr}']
            m = float(np.mean([r['mean_pct'] for r in runs]))
            l = float(np.mean([r['late_mean_pct'] for r in runs]))
            out['grid'][f'{nd}'][f'{fr}'] = {'mean_pct': m, 'late_mean_pct': l}
            row.append(f'{m:6.2f}% ({l:5.2f}%)')
        print(f'{nd:11.3f} ' + ' '.join(row))
    # break-even: largest noise at which phase still beats tau-at-deployment
    be = [nd for nd in NOISE_DEX
          if out['grid'][f'{nd}']['0.0']['mean_pct'] < 6.975]
    out['break_even_noise_dex'] = max(be) if be else None
    print(f"\nphase-conditioning beats tau-at-deployment (6.98%) up to "
          f"{out['break_even_noise_dex']} dex of central-density noise")
    json.dump(out, open(os.path.join(OUT, 'phase_noise.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
