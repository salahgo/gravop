"""E24 — train the conditioning channels on the inputs they will actually see.

E23 found that joint (time + state) conditioning is the one hybrid that helps,
and read it as sensor fusion: tau and phi are two noisy estimates of the same
latent quantity ("how far along is this evolution?"), with uncorrelated errors.
tau's error is a systematic phase shift inherited from mispredicting t_sing,
largest near the singularity; phi's is measurement noise on a field, unrelated
to proximity to the singularity.

But E23 trained every conditioning channel on its EXACT value and evaluated on
a corrupted one:

    train:  cond = (params, tau_true,     phi_true,  flag_true)
    deploy: cond = (params, tau_from_head, phi+noise, flag+flips)

That is a train/test mismatch on both channels, and it biases the comparison in
a specific direction: a model shown an exact tau during training learns to trust
it, then meets a corrupted tau at deployment. It explains why H1 was *worse*
than the state-only model at zero phase noise (sidm 0.0169 vs 0.0043) despite
being strictly better informed. Every E23 noise number is therefore the
pessimistic "train clean, deploy dirty" case -- for the state-only model too.

This experiment removes the mismatch. Arms:

  B_aug   time-conditioned only, trained with tau perturbed per example. THE
          CONTROL, and the reason the fusion reading can be tested rather than
          asserted: a joint model beating arm B could be winning from the
          augmentation alone. It also re-measures the E22 deployment penalty
          under a training recipe that does not hand the model an exact tau, so
          it says how much of that headline was a training artefact.
  C_aug   state-conditioned, trained with phi perturbed per example
  H1_aug  joint, trained with BOTH channels perturbed per example
  H2_rel  joint + reliability: the per-example noise scales (sigma_tau,
          sigma_phi) are themselves fed to the operator, so it can weight the
          two channels by how good each is on this particular query rather than
          learning one fixed compromise. This is the proper fusion form, and it
          is available whenever a user knows their own measurement quality.

Augmentation distributions are sampled per example so one model covers the whole
noise range rather than being tuned to a single operating point:
    sigma_phi ~ U[0, PHI_MAX] dex,  flag flipped w.p. q ~ U[0, Q_MAX]
    sigma_tau ~ U[0, TAU_MAX] relative, applied as tau / (1 + N(0, sigma_tau))
TAU_MAX spans the 1.8-7.1% t_sing errors the trained heads actually achieve.

Usage:  sc_robust.py --system rd [--seeds 3] [--device cuda:0]
Output: results/statecond/robust_<system>.json
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
from statecond import data as scd                                  # noqa: E402
from statecond.model import OperatorNet                            # noqa: E402
from sc_experiment import (Norm, tensors, train_operator,          # noqa: E402
                           train_time_head, LATE, OUT)
from sc_hybrid import EPS_GRID, Q_GRID                             # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
PHI_MAX = 0.20      # dex, matches the top of the E23 sweep
Q_MAX = 0.30        # branch-flip probability, matches the top of the E23 sweep
TAU_MAX = 0.15      # relative t_sing error; heads achieve 0.018-0.071


def cond_dim_for(kind):
    """params + conditioning channels."""
    return {'time_aug': 1, 'phase_aug': 2, 'joint_aug': 3, 'joint_rel': 5}[kind]


def build_cond(kind, par, ri, taus_t, ti, phi_v, flag_v, s_tau, s_phi):
    """Assemble the conditioning vector for one batch."""
    parts = [par[ri]]
    if kind != 'phase_aug':
        parts.append(taus_t[:, None])
    if kind != 'time_aug':
        parts += [phi_v[:, None], flag_v[:, None]]
    if kind == 'joint_rel':
        parts += [s_tau[:, None], s_phi[:, None]]
    return torch.cat(parts, 1)


def train_robust(kind, T, tr, seed, dev, epochs, n_fields, n_par, phi_sd):
    """Train with per-example corruption of the conditioning channels."""
    u, u0, par, phi, flag, _ = T
    torch.manual_seed(seed + 991)
    np.random.seed(seed + 991)
    g = torch.Generator(device=dev).manual_seed(seed + 991)
    taus = torch.tensor(scd.tau_grid(), dtype=torch.float32, device=dev)
    n_tau = u.shape[1]

    net = OperatorNet(n_fields=n_fields, n_par=n_par,
                      cond_extra=cond_dim_for(kind), n_x=u.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ri_all = np.repeat(tr, n_tau)
    ti_all = np.tile(np.arange(n_tau), len(tr))
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(len(ri_all))
        for s in range(0, len(perm), 1024):
            sel = perm[s:s + 1024]
            ri, ti = ri_all[sel], ti_all[sel]
            b = len(sel)

            # per-example noise scales, drawn fresh every batch
            s_phi = torch.rand(b, device=dev, generator=g) * PHI_MAX
            q = torch.rand(b, device=dev, generator=g) * Q_MAX
            s_tau = torch.rand(b, device=dev, generator=g) * TAU_MAX

            # phase channel: additive noise in dex -> normalised units
            noise = torch.randn(b, device=dev, generator=g) * (s_phi / phi_sd)
            phi_v = phi[ri, ti] + noise
            flip = (torch.rand(b, device=dev, generator=g) < q).float()
            flag_v = torch.abs(flag[ri, ti] - flip)

            # time channel: a mispredicted t_sing rescales tau
            delta = torch.randn(b, device=dev, generator=g) * s_tau
            taus_t = torch.clamp(taus[ti] / (1.0 + delta), 0.0, 1.0)

            cond = build_cond(kind, par, ri, taus_t, ti, phi_v, flag_v,
                              s_tau, s_phi)
            opt.zero_grad()
            nn.functional.mse_loss(net(u0[ri], cond), u[ri, ti]).backward()
            opt.step()
        sch.step()
    return net


@torch.no_grad()
def err_robust(kind, net, T, te, tau_dep, phi_obs, flag_obs, s_tau, s_phi):
    u, u0, par, _, _, _ = T
    dev = u.device
    n_tau = phi_obs.shape[1]
    ri = np.asarray(te)
    out = np.zeros((len(te), n_tau))
    st = torch.full((len(te),), float(s_tau), device=dev)
    sp = torch.full((len(te),), float(s_phi), device=dev)
    for j in range(n_tau):
        tq = torch.tensor(tau_dep[:, j], dtype=torch.float32, device=dev)
        cond = build_cond(kind, par, ri, tq, j, phi_obs[:, j], flag_obs[:, j],
                          st, sp)
        out[:, j] = (net(u0[ri], cond) - u[ri, j]).pow(2).mean(
            dim=(1, 2)).sqrt().cpu().numpy()
    return out


def merge_into(path, res):
    """Fold newly computed arms into an existing result file.

    The noise realisations are reproducible -- `rng` is re-seeded per run() and
    consumed in a fixed order, and the time head is seeded from the run seed --
    so arms computed in a later pass see the same corrupted inputs as the
    original ones and the columns stay comparable.
    """
    old = json.load(open(path))
    key = lambda r: (r['seed'], r['eps_dex'], r['q'])          # noqa: E731
    new_rows = {key(r): r for r in res['rows']}
    for r in old['rows']:
        r.update({k: v for k, v in new_rows.get(key(r), {}).items()
                  if k not in ('seed', 'eps_dex', 'q')})
    for cfg, vals in res['aggregate'].items():
        old['aggregate'].setdefault(cfg, {}).update(vals)
    old['aug'] = res['aug']
    return old


def run(name, seeds, dev, epochs, data_dir, kinds=None):
    d = scd.load(os.path.join(data_dir, f'{name}.h5'))
    n_runs, n_tau, n_fields, n_x = d['u'].shape
    n_par = d['par'].shape[1]
    tr, va, te = scd.split(n_runs)
    n = Norm(d, tr)
    T = tensors(d, n, dev)
    u, u0, par, phi, flag, lts = T
    taus_np = scd.tau_grid()
    late = taus_np > LATE
    rng = np.random.default_rng(20260830)
    # time_aug is the control that keeps the joint result honest: it is
    # time-conditioned only, but trained with the same tau corruption. Without
    # it, a joint model beating arm B could be winning from the augmentation
    # alone rather than from combining the two channels.
    full_pass = kinds is None
    if full_pass:
        kinds = ['time_aug', 'phase_aug', 'joint_aug', 'joint_rel']

    rows = []
    for s in seeds:
        head = train_time_head(T, tr, s, dev, epochs, n_fields, n_par)
        with torch.no_grad():
            lt_pred = head(u0[te], par[te]).cpu().numpy() * n.t_sd + n.t_mu
        ratio = 10.0 ** (lt_pred - d['log_tsing'][te])
        tau_dep = np.clip(np.tile(taus_np, (len(te), 1)) / ratio[:, None], 0, 1)
        head_err = float(np.median(np.abs(ratio - 1)))

        # clean-trained reference, identical to E22/E23; skipped when this is
        # a merge pass that only adds a control arm
        # only the full default pass computes it; a targeted --kinds pass is a
        # merge that reuses the reference already in the result file
        net_c = (train_operator('phase', T, tr, te, s, dev, epochs,
                                n_fields, n_par)[0] if full_pass else None)
        nets = {k: train_robust(k, T, tr, s, dev, epochs, n_fields, n_par,
                                n.phi_sd) for k in kinds}
        print(f"  [{name}] trained seed {s}", flush=True)

        for eps, q in ([(e, 0.0) for e in EPS_GRID]
                       + [(0.0, qq) for qq in Q_GRID if qq > 0]):
            eps_n = eps / n.phi_sd
            noise = torch.tensor(
                rng.normal(0.0, eps_n, size=tuple(phi[te].shape)),
                dtype=torch.float32, device=dev)
            flips = torch.tensor(
                (rng.random(tuple(flag[te].shape)) < q).astype(np.float32),
                device=dev)
            phi_o = phi[te] + noise
            flag_o = torch.abs(flag[te] - flips)

            row = {'seed': s, 'eps_dex': eps, 'q': q}
            # the clean-trained state-conditioned model, carried through as the
            # E23 reference; it takes (phi, flag) only, so it needs its own loop
            if net_c is not None:
                with torch.no_grad():
                    e = np.zeros((len(te), n_tau))
                    ri = np.asarray(te)
                    for j in range(n_tau):
                        cond = torch.cat([par[ri], phi_o[:, j][:, None],
                                          flag_o[:, j][:, None]], 1)
                        e[:, j] = (net_c(u0[ri], cond) - u[ri, j]).pow(2).mean(
                            dim=(1, 2)).sqrt().cpu().numpy()
                row['C_clean_trained'] = float(e.mean())
                row['C_clean_trained_late'] = float(e[:, late].mean())

            for k in kinds:
                ek = err_robust(k, nets[k], T, te, tau_dep, phi_o, flag_o,
                                head_err, eps)
                row[k] = float(ek.mean())
                row[k + '_late'] = float(ek[:, late].mean())
            rows.append(row)
            print(f"  [{name} s{s}] eps={eps:.2f} q={q:.2f} | "
                  + '  '.join(f'{k} {row[k]:.4f}' for k in row
                              if k not in ('seed', 'eps_dex', 'q')
                              and not k.endswith('_late')), flush=True)

    keys = [k for k in rows[0] if k not in ('seed', 'eps_dex', 'q')]
    agg = {}
    for eps, q in ([(e, 0.0) for e in EPS_GRID]
                   + [(0.0, qq) for qq in Q_GRID if qq > 0]):
        sel = [r for r in rows if r['eps_dex'] == eps and r['q'] == q]
        agg[f'eps{eps}_q{q}'] = {k: {'mean': float(np.mean([r[k] for r in sel])),
                                     'std': float(np.std([r[k] for r in sel]))}
                                 for k in keys}
    return {'system': name, 'seeds': list(seeds), 'epochs': epochs,
            'n_test': int(len(te)), 'phi_sd_dex': float(n.phi_sd),
            'aug': {'phi_max_dex': PHI_MAX, 'q_max': Q_MAX,
                    'tau_max_rel': TAU_MAX},
            'eps_grid_dex': EPS_GRID, 'q_grid': Q_GRID,
            'rows': rows, 'aggregate': agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'sc'))
    ap.add_argument('--kinds', default=None,
                    help='comma-separated subset of arms to train')
    ap.add_argument('--merge', action='store_true',
                    help='fold the computed arms into an existing result file')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(args.device)
    from statecond.pdes import SYSTEMS
    names = list(SYSTEMS) + ['sidm'] if args.all else [args.system]
    for name in names:
        print(f"=== robust {name} ===", flush=True)
        kinds = args.kinds.split(',') if args.kinds else None
        res = run(name, range(args.seeds), dev, args.epochs, args.data, kinds)
        path = os.path.join(OUT, f'robust_{name}.json')
        if args.merge and os.path.exists(path):
            res = merge_into(path, res)
        json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
