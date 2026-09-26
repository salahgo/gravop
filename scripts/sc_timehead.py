"""P20 -- is the deployment penalty partly a training artefact? (2026-09-05)

WHAT PROMPTED THIS
------------------
P14 trained a clock head with the same architecture as `TimeHead`, the same
inputs (u0, par), the same split -- and, incidentally, 22x more optimiser steps,
because it iterates over (trajectory, tau) pairs where `train_time_head`
iterates over trajectories alone. It reached 0.30% median relative error on
`rd`, where the published head reports 7.11%.

That is not a better model. It is the same model, better trained. And the
deployment penalty this paper reports -- arm B against arm A -- is entirely a
function of how good that head is, so if the head is undertrained then so is
the headline.

WHAT THIS MEASURES
------------------
1. The head's test error as a function of optimiser steps, per system, to
   distinguish "undertrained" from "we happened to pick a lucky budget". A
   converged curve is the evidence; one better number is not.
2. Arm B recomputed with the converged head, from the PUBLISHED operator
   checkpoints -- arm B is arm A's operator queried at a wrong tau, so no
   operator is retrained and the only thing that changes is the clock.
3. The consequences: the deployment penalty B/A, and arm C's advantage B/C.

Output: results/statecond/timehead.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import statecond.data as scd                                    # noqa: E402
from statecond.model import TimeHead                             # noqa: E402
from sc_experiment import (LATE, Norm, field_err, load_checkpoint,  # noqa: E402
                           save_checkpoint, tensors)
from sc_endpoint import check_plain_arm                           # noqa: E402
from sc_provenance import stamp                                  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(OUT, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']
# the published recipe is (200 epochs, batch 256); the sweep holds the batch
# size and varies only the number of passes, so nothing but the budget changes
BUDGETS = [200, 600, 2000, 6000, 20000]


def train_head(T, tr, seed, dev, epochs, n_fields, n_par, batch=256):
    """`sc_experiment.train_time_head`, with the epoch count exposed."""
    u, u0, par, _, _, lts = T
    torch.manual_seed(seed + 777)
    head = TimeHead(n_fields=n_fields, n_par=n_par, n_x=u.shape[-1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_t = torch.tensor(np.asarray(tr), device=dev)
    for _ in range(epochs):
        head.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for s in range(0, len(perm), batch):
            sel = perm[s:s + batch]
            opt.zero_grad()
            nn.functional.mse_loss(head(u0[sel], par[sel]),
                                   lts[sel]).backward()
            opt.step()
        sch.step()
    return head.eval()


def head_error(head, T, te, n, log_tsing, dev):
    u, u0, par, _, _, _ = T
    with torch.no_grad():
        lp = head(u0[te], par[te]).cpu().numpy() * n.t_sd + n.t_mu
    ratio = 10.0 ** (lp - log_tsing[te])
    return float(100 * np.median(np.abs(ratio - 1))), ratio


def load_converged_head(name, seed, dev, backbone='fno'):
    """The converged clock as a saved checkpoint, or (None, None) if absent.

    `--save-heads` writes these; if they are missing the caller must say so
    rather than silently falling back to the published head, which is the
    exact substitution P20 showed changes every downstream number.
    """
    p = os.path.join(CKPT, f'{name}_{backbone}_time_conv_seed{seed}.pt')
    if not os.path.exists(p):
        return None, None
    return load_checkpoint(p, dev)


def converged_armB(name, T, te, n, log_tsing, dev, seeds=3, backbone='fno'):
    """Arm B evaluated through the CONVERGED clock, from saved checkpoints.

    P20 measured this inside `main`'s own loop. P22 needs the same quantity as
    a *comparator* inside `sc_contract`, and re-deriving it there would be both
    slow and a second implementation to keep in step -- which is precisely the
    failure that made v7 necessary (P13 was written against arm D because arm B
    through a converged clock did not yet exist as a callable thing). This is
    the one implementation; both callers use it.

    Returns (mean nRMSE, per-trajectory array of shape (seeds, n_test)), or
    (None, None) if the converged heads were never saved.
    """
    u, u0, par, phi, flag, _ = T
    taus = scd.tau_grid()
    taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)
    per_seed = []
    for s in range(seeds):
        head, _ = load_converged_head(name, s, dev, backbone)
        if head is None:
            return None, None
        ptau = os.path.join(CKPT, f'{name}_{backbone}_tau_seed{s}.pt')
        net_a, ma = load_checkpoint(ptau, dev)
        check_plain_arm(ma, ptau)

        def cond_a(ri, ti):
            return torch.cat([par[ri], taus_t[ti][:, None]], 1)

        with torch.no_grad():
            lp = head(u0[te], par[te]).cpu().numpy() * n.t_sd + n.t_mu
        ratio = 10.0 ** (lp - log_tsing[te])
        tq = np.clip(np.tile(taus, (len(te), 1)) / ratio[:, None], 0, 1)
        e = field_err(net_a, cond_a, u, u0, te, tau_query=tq)
        per_seed.append(e.mean(axis=1))          # per test trajectory
        del net_a, head
        torch.cuda.empty_cache()
    per_seed = np.asarray(per_seed)
    return float(per_seed.mean()), per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--system', default=None)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--budgets', type=int, nargs='*', default=BUDGETS)
    ap.add_argument('--save-heads', action='store_true',
                    help='write the converged heads as <sys>_fno_time_conv_'
                         '<seed>.pt so sc_displacement can be re-run against them')
    args = ap.parse_args()
    dev = args.device

    path = os.path.join(OUT, 'timehead.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    names = SYSTEMS if args.all else [args.system]

    for name in names:
        d = scd.load(os.path.join(DATA, f'{name}.h5'))
        n_runs, n_tau, n_fields, n_x = d['u'].shape
        n_par = d['par'].shape[1]
        tr, va, te = scd.split(n_runs)
        n = Norm(d, tr)
        T = tensors(d, n, dev)
        u, u0, par, phi, flag, lts = T
        taus = scd.tau_grid()
        late = taus > LATE
        steps_per_epoch = int(np.ceil(len(tr) / 256))

        curve, ratios_at = {}, {}
        for ep in args.budgets:
            errs, rr = [], []
            for s in range(args.seeds):
                h = train_head(T, tr, s, dev, ep, n_fields, n_par)
                e, ratio = head_error(h, T, te, n, d['log_tsing'], dev)
                errs.append(e)
                rr.append(ratio)
                if args.save_heads and ep == max(args.budgets):
                    os.makedirs(CKPT, exist_ok=True)
                    save_checkpoint(
                        os.path.join(CKPT,
                                     f'{name}_fno_time_conv_seed{s}.pt'),
                        h, 'time', 'fno', n, s, ep, (n_fields, n_par, n_x))
                del h
                torch.cuda.empty_cache()
            curve[ep] = {'mean': float(np.mean(errs)),
                         'std': float(np.std(errs)),
                         'steps': ep * steps_per_epoch}
            ratios_at[ep] = rr
            print(f'  {name} {ep:6d} ep ({ep*steps_per_epoch:7d} steps): '
                  f'{np.mean(errs):6.3f}%', flush=True)

        # NOT argmin over the curve: picking the budget that scores best on
        # the test split is test-set selection, however mild. The rule is
        # "the largest budget", fixed in advance. The sweep is evidence that
        # the curve has flattened by then, not a menu to choose from.
        best_ep = max(curve)

        # arm B recomputed with the converged clock, from published weights
        armB, armB_late, armA, armC = [], [], [], []
        # P23: per-trajectory vectors, shape (seeds, n_test), so the converged
        # penalties -- which are the smallest effects in the project, 0.97 in
        # `burgers` and 1.06 in `sidm` -- can carry a paired trajectory
        # bootstrap instead of a three-seed spread.
        traj = {'A': [], 'B': [], 'C': []}
        for s in range(args.seeds):
            ptau = os.path.join(CKPT, f'{name}_fno_tau_seed{s}.pt')
            ppha = os.path.join(CKPT, f'{name}_fno_phase_seed{s}.pt')
            net_a, ma = load_checkpoint(ptau, dev)
            net_c, mc = load_checkpoint(ppha, dev)
            check_plain_arm(ma, ptau)
            check_plain_arm(mc, ppha)
            taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)

            def cond_a(ri, ti):
                return torch.cat([par[ri], taus_t[ti][:, None]], 1)

            def cond_c(ri, ti):
                return torch.cat([par[ri], phi[ri, ti][:, None],
                                  flag[ri, ti][:, None]], 1)

            ratio = ratios_at[best_ep][s]
            tq = np.clip(np.tile(taus, (len(te), 1)) / ratio[:, None], 0, 1)
            e = field_err(net_a, cond_a, u, u0, te, tau_query=tq)
            armB.append(float(e.mean()))
            armB_late.append(float(e[:, late].mean()))
            traj['B'].append(e.mean(axis=1))
            eA = field_err(net_a, cond_a, u, u0, te)
            eC = field_err(net_c, cond_c, u, u0, te)
            armA.append(float(eA.mean()))
            armC.append(float(eC.mean()))
            traj['A'].append(eA.mean(axis=1))
            traj['C'].append(eC.mean(axis=1))
            del net_a, net_c
            torch.cuda.empty_cache()

        pub = json.load(open(os.path.join(OUT, f'{name}.json')))
        rec = {
            'system': name,
            'seeds': args.seeds,
            'steps_per_epoch': steps_per_epoch,
            'published_recipe_epochs': 200,
            'published_head_err_pct': pub['tsing_median_err_pct']['mean'],
            'curve': {str(k): v for k, v in curve.items()},
            'converged_epochs': best_ep,
            'budget_rule': 'largest budget swept, fixed in advance',
            'converged_head_err_pct': curve[best_ep]['mean'],
            'head_improvement': (pub['tsing_median_err_pct']['mean']
                                 / curve[best_ep]['mean']),
            'armA': float(np.mean(armA)),
            'armC': float(np.mean(armC)),
            'armB_converged': float(np.mean(armB)),
            'armB_converged_late': float(np.mean(armB_late)),
            'armB_published': pub['nrmse']['B_tau_deployed']['mean'],
            'penalty_published': pub['deployment_penalty'],
            'penalty_converged': float(np.mean(armB)) / float(np.mean(armA)),
            'state_vs_deployed_published': pub['state_vs_deployed'],
            'state_vs_deployed_converged': (float(np.mean(armB))
                                            / float(np.mean(armC))),
            'per_trajectory': {k: np.asarray(v).tolist()
                               for k, v in traj.items()},
            'test_index': np.asarray(te).tolist(),
            '_provenance': stamp('sc_timehead.py'),
        }
        out[name] = rec
        print(f"{name:8s} head {rec['published_head_err_pct']:.2f}% -> "
              f"{rec['converged_head_err_pct']:.3f}% "
              f"({rec['head_improvement']:.1f}x) | penalty "
              f"{rec['penalty_published']:.2f}x -> "
              f"{rec['penalty_converged']:.2f}x | state-vs-deployed "
              f"{rec['state_vs_deployed_published']:.2f}x -> "
              f"{rec['state_vs_deployed_converged']:.2f}x", flush=True)
        json.dump(out, open(path, 'w'), indent=1)

    done = [k for k in SYSTEMS if k in out]
    if len(done) == len(SYSTEMS):
        out['_summary'] = {
            'finding': ('the published time head is undertrained: 200 epochs '
                        'over trajectories is a few hundred optimiser steps'),
            'head_improvement_range': [
                min(out[k]['head_improvement'] for k in done),
                max(out[k]['head_improvement'] for k in done)],
            'penalty_published_range': [
                min(out[k]['penalty_published'] for k in done),
                max(out[k]['penalty_published'] for k in done)],
            'penalty_converged_range': [
                min(out[k]['penalty_converged'] for k in done),
                max(out[k]['penalty_converged'] for k in done)],
            'state_still_wins_in': [
                k for k in done if out[k]['state_vs_deployed_converged'] > 1.0],
        }
        json.dump(out, open(path, 'w'), indent=1)
        print('\npenalty range', out['_summary']['penalty_published_range'],
              '->', out['_summary']['penalty_converged_range'])
        print('state still beats deployed time in:',
              out['_summary']['state_still_wins_in'])
    print('wrote', path)


if __name__ == '__main__':
    main()
