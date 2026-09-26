"""E25 — the baseline the whole framing has to beat: autoregressive rollout.

A literature check (2026-08-30) confirms an earlier novelty assessment and refutes the
"conventional" wording that had crept into the reframing: for time-dependent
PDE surrogates the standard approach is *autoregressive rollout*, in which the
operator maps the current state to the next one and is applied recursively
(arXiv:2605.25413, arXiv:2602.18801, arXiv:2506.06158 all describe it that way
while proposing alternatives to it). Time-rescaling by a singular time does
appear in the singularity-formation literature (arXiv:2604.16842,
arXiv:2501.07377), but there it is used to *characterise* a blow-up whose time
is being computed, not to build a surrogate whose blow-up time must be predicted.

That makes rollout a structural competitor rather than one more baseline:
it needs no `t_sing` and no phase variable at all. If it is accurate and cheap,
the conditioning question this project is built on does not need answering.

Arms, on the same data, splits, seeds, backbone and budget as E22:

  R_naive  teacher-forced, predicts the next state directly. Maps
           (params, u_k) -> u_{k+1} with NO timing input, rolled from u_0.
           Included only to show why the stabilisers below are needed: over 47
           steps it diverges outright on rd and film.
  R_res    predicts the *residual* u_{k+1} - u_k, and is trained with Gaussian
           noise on the input state. Residual parameterisation and input-noise
           injection are the two standard stabilisers for autoregressive PDE
           surrogates; E24's lesson was the same one -- at rollout time the
           input is the model's own imperfect output, not a ground-truth field.
  R_push   R_res plus pushforward training: the loss is taken on the second of
           two unrolled steps, with the gradient blocked through the first, so
           the model sees its own error distribution during training. This is
           the strongest form we know how to build, and it is the arm the
           conclusion must be defended against.

Making the baseline strong is the point. A rollout that is allowed to diverge
proves nothing, and a referee would rightly reject the comparison.

Reported: nRMSE per grid point (the error-accumulation curve), the mean and
late-time aggregates comparable with arms A/B/C, and the sequential cost.

Note on the grid. Snapshots sit on the tau grid, which is refined past
tau = 0.92, so the physical step size changes partway through and the model is
never told it. That is a real handicap and it is quantified rather than hidden:
the per-step curve is reported, and the aggregate is also given restricted to
the uniform part of the grid.

Usage:  sc_rollout.py --system rd [--seeds 3] [--device cuda:0]
Output: results/statecond/rollout_<system>.json
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
from statecond import data as scd                                  # noqa: E402
from statecond.model import OperatorNet                            # noqa: E402
from sc_experiment import Norm, tensors, LATE, OUT                 # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
NOISE_SD = 0.02      # input-state noise for R_noise, in normalised field units


def step(net, x, p, residual):
    """One rollout step, in whichever parameterisation is being used."""
    return x + net(x, p) if residual else net(x, p)


def train_rollout(T, tr, seed, dev, epochs, n_fields, n_par,
                  noise=0.0, residual=False, push=False):
    """One-step operator: (params, u_k) -> u_{k+1}. No timing input."""
    u, u0, par, _, _, _ = T
    torch.manual_seed(seed + 4242)
    np.random.seed(seed + 4242)
    g = torch.Generator(device=dev).manual_seed(seed + 4242)
    n_tau = u.shape[1]

    net = OperatorNet(n_fields=n_fields, n_par=n_par, cond_extra=0,
                      n_x=u.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # pairs (k -> k+1); pushforward needs k+2 to exist as well
    last = n_tau - 2 if push else n_tau - 1
    ri_all = np.repeat(tr, last)
    ki_all = np.tile(np.arange(last), len(tr))
    t0 = time.time()
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(len(ri_all))
        for s in range(0, len(perm), 1024):
            sel = perm[s:s + 1024]
            ri, ki = ri_all[sel], ki_all[sel]
            x = u[ri, ki]
            if noise > 0:
                x = x + noise * torch.randn(x.shape, device=dev, generator=g)
            opt.zero_grad()
            if push:
                # one step with no gradient, so the second step is taken from
                # the model's own (imperfect) output; loss on that second step
                with torch.no_grad():
                    x1 = step(net, x, par[ri], residual)
                pred = step(net, x1, par[ri], residual)
                tgt = u[ri, ki + 2]
            else:
                pred = step(net, x, par[ri], residual)
                tgt = u[ri, ki + 1]
            nn.functional.mse_loss(pred, tgt).backward()
            opt.step()
        sch.step()
    return net, time.time() - t0


@torch.no_grad()
def roll(net, T, te, residual):
    """Roll from u_0 and score every grid point. Returns (err, wall_seconds).

    Predictions are clamped to a generous multiple of the observed field range;
    without it a diverging rollout returns inf and destroys the aggregate rather
    than reporting a large but finite error.
    """
    u, u0, par, _, _, _ = T
    n_tau = u.shape[1]
    ri = np.asarray(te)
    lim = float(u.abs().max()) * 10.0
    err = np.zeros((len(te), n_tau))
    x = u0[ri].clone()
    t0 = time.time()
    for k in range(1, n_tau):
        x = torch.nan_to_num(step(net, x, par[ri], residual),
                             nan=0.0, posinf=lim, neginf=-lim).clamp(-lim, lim)
        err[:, k] = (x - u[ri, k]).pow(2).mean(dim=(1, 2)).sqrt().cpu().numpy()
    if u.is_cuda:
        torch.cuda.synchronize()
    return err, time.time() - t0


@torch.no_grad()
def direct_cost(net, T, te):
    """Cost of a single random-access query, for the comparison that matters to
    MCMC: a rollout cannot answer 'the field at one time' without computing
    every earlier step."""
    u, u0, par, _, _, _ = T
    ri = np.asarray(te)
    t0 = time.time()
    net(u0[ri], par[ri])
    torch.cuda.synchronize() if u.is_cuda else None
    return time.time() - t0


def run(name, seeds, dev, epochs, data_dir):
    d = scd.load(os.path.join(data_dir, f'{name}.h5'))
    n_runs, n_tau, n_fields, n_x = d['u'].shape
    n_par = d['par'].shape[1]
    tr, va, te = scd.split(n_runs)
    n = Norm(d, tr)
    T = tensors(d, n, dev)
    taus = scd.tau_grid()
    late = taus > LATE
    uniform = np.arange(n_tau) < 24        # the un-refined part of the grid

    res = {'system': name, 'seeds': list(seeds), 'epochs': epochs,
           'n_test': int(len(te)), 'noise_sd': NOISE_SD,
           'tau_grid': taus.tolist(), 'arms': {}}
    arms = (('R_naive', dict(noise=0.0, residual=False, push=False)),
            ('R_res', dict(noise=NOISE_SD, residual=True, push=False)),
            ('R_push', dict(noise=NOISE_SD, residual=True, push=True)))
    for lab, cfg in arms:
        means, lates, unis, curves, fits, rolls, singles = [], [], [], [], [], [], []
        for s in seeds:
            net, fit = train_rollout(T, tr, s, dev, epochs, n_fields, n_par,
                                     **cfg)
            err, wall = roll(net, T, te, cfg['residual'])
            means.append(float(err[:, 1:].mean()))
            lates.append(float(err[:, late].mean()))
            unis.append(float(err[:, uniform & (np.arange(n_tau) > 0)].mean()))
            curves.append(err.mean(axis=0).tolist())
            fits.append(fit)
            rolls.append(wall)
            singles.append(direct_cost(net, T, te))
            print(f"  [{name} {lab} s{s}] mean {means[-1]:.4f}  "
                  f"late {lates[-1]:.4f}  uniform-grid {unis[-1]:.4f}",
                  flush=True)
        res['arms'][lab] = {
            'config': cfg,
            'mean': {'mean': float(np.mean(means)), 'std': float(np.std(means))},
            'late': {'mean': float(np.mean(lates)), 'std': float(np.std(lates))},
            'uniform_grid': {'mean': float(np.mean(unis)),
                             'std': float(np.std(unis))},
            'curve': np.mean(curves, axis=0).tolist(),
            'fit_seconds': float(np.mean(fits)),
            'rollout_seconds_full_trajectory': float(np.mean(rolls)),
            'seconds_single_query': float(np.mean(singles)),
            'sequential_steps_per_query': int(n_tau - 1),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'sc'))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(args.device)
    from statecond.pdes import SYSTEMS
    names = list(SYSTEMS) + ['sidm'] if args.all else [args.system]
    for name in names:
        print(f"=== rollout {name} ===", flush=True)
        res = run(name, range(args.seeds), dev, args.epochs, args.data)
        json.dump(res, open(os.path.join(OUT, f'rollout_{name}.json'), 'w'),
                  indent=1)


if __name__ == '__main__':
    main()
