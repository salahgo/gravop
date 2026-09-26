"""Hybrid conditioning schemes under imperfect phase measurement.

State-conditioning wins when the state coordinate is measured well and loses
when it is not (E21: the gravothermal advantage survives to ~0.05 dex of
central-density noise and a 15% branch-misidentification rate alone costs
0.64% -> 2.60%). Its weakness is therefore specific and known, and it points at
a specific fix: stop conditioning on a *point estimate* of the phase.

Schemes compared, at matched training budget and on the same splits:

  B   tau-conditioned, deployed        t_sing from the learned head. Carries no
                                       dependence on the phase measurement, so
                                       it is the flat reference line.
  C   state-conditioned, point         phi plugged in as measured. The E20/E22
                                       method; degrades as noise rises.
  H1  joint (tau, phi)                 both scalars supplied, both imperfect.
                                       Tests whether the operator can learn to
                                       fall back on whichever is more reliable.
  H2  marginalised state               integrate the operator against the phase
                                       posterior instead of evaluating it at
                                       the posterior mean:
                                         y = sum_b P(b) int f(y|phi,b) p(phi) dphi
                                       Gauss-Hermite in phi, exact sum over the
                                       two branches. K extra forward passes, no
                                       extra training. Also yields a predictive
                                       spread, which C does not have.
  H3  ensemble state                   the per-seed state-conditioned operators
                                       averaged. This is the control that
                                       separates "averaging several predictions
                                       helps" from "averaging over the uncertain
                                       variable helps": H2 and H3 both average
                                       K forward passes, but only H2 varies the
                                       phase between them. The ensemble members
                                       are the same networks reported as C at
                                       each seed, so H3 is not independent of C
                                       -- it is a deep ensemble in the usual
                                       sense, not a fresh draw.

Noise model, following E21: the measured phase is phi + N(0, eps) in dex, and
the branch flag is flipped with probability q. eps is swept with q = 0, and q
is probed separately at eps = 0, so the two failure channels stay separable.

A fourth scheme is measured separately: whether the phase can be inverted back
into a *time*. If the observed state pins down where in the evolution the
system is, it also pins down t_sing, which is what a tau-conditioned user
actually wants. `state_to_time` reports that.

Usage:  sc_hybrid.py --system rd [--seeds 3] [--device cuda:0]
Output: results/statecond/hybrid_<system>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from statecond import data as scd                                  # noqa: E402
from statecond.model import OperatorNet                            # noqa: E402
from sc_experiment import (Norm, tensors, train_operator,           # noqa: E402
                           train_time_head, LATE, OUT)

ROOT = os.path.join(os.path.dirname(__file__), '..')
EPS_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
Q_GRID = [0.0, 0.05, 0.15, 0.30]
GH_NODES = 7


def train_joint(T, tr, seed, dev, epochs, n_fields, n_par):
    """H1: operator conditioned on parameters, tau AND (phi, flag)."""
    import torch.nn as nn
    u, u0, par, phi, flag, _ = T
    torch.manual_seed(seed + 313)
    np.random.seed(seed + 313)
    taus = torch.tensor(scd.tau_grid(), dtype=torch.float32, device=dev)
    n_tau = u.shape[1]
    net = OperatorNet(n_fields=n_fields, n_par=n_par, cond_extra=3,
                      n_x=u.shape[-1]).to(dev)
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
            cond = torch.cat([par[ri], taus[ti][:, None],
                              phi[ri, ti][:, None], flag[ri, ti][:, None]], 1)
            opt.zero_grad()
            nn.functional.mse_loss(net(u0[ri], cond), u[ri, ti]).backward()
            opt.step()
        sch.step()
    return net


@torch.no_grad()
def err_state(nets, u, u0, par, phi_obs, flag_obs, te, marginal=None,
              return_spread=False):
    """nRMSE for state-conditioned operators.

    nets       list of operators (len 1 = point estimate, >1 = ensemble)
    phi_obs    (n_te, n_tau) observed phase, already normalised
    marginal   None, or (eps_norm, q): integrate over the phase posterior
    """
    n_tau = phi_obs.shape[1]
    out = np.zeros((len(te), n_tau))
    spread = np.zeros((len(te), n_tau))
    ri = np.asarray(te)
    if marginal is None:
        nodes, wts = np.array([0.0]), np.array([1.0])
        q = 0.0
    else:
        eps_n, q = marginal
        h, w = np.polynomial.hermite_e.hermegauss(GH_NODES)
        nodes, wts = h * eps_n, w / w.sum()
    for j in range(n_tau):
        acc, acc2, wsum = 0.0, 0.0, 0.0
        for node, w in zip(nodes, wts):
            p_j = phi_obs[:, j] + float(node)
            for b, wb in ((flag_obs[:, j], 1.0 - q), (1.0 - flag_obs[:, j], q)):
                if wb <= 0:
                    continue
                cond = torch.cat([par[ri], p_j[:, None], b[:, None]], 1)
                pred = torch.stack([n(u0[ri], cond) for n in nets]).mean(0)
                acc = acc + w * wb * pred
                acc2 = acc2 + w * wb * pred.pow(2)
                wsum += w * wb
        pred = acc / wsum
        var = (acc2 / wsum - pred.pow(2)).clamp_min(0)
        out[:, j] = (pred - u[ri, j]).pow(2).mean(dim=(1, 2)).sqrt().cpu().numpy()
        spread[:, j] = var.mean(dim=(1, 2)).sqrt().cpu().numpy()
    return (out, spread) if return_spread else out


@torch.no_grad()
def err_joint(net, u, u0, par, tau_q, phi_obs, flag_obs, te):
    n_tau = phi_obs.shape[1]
    out = np.zeros((len(te), n_tau))
    ri = np.asarray(te)
    for j in range(n_tau):
        tq = torch.tensor(tau_q[:, j], dtype=torch.float32, device=u.device)
        cond = torch.cat([par[ri], tq[:, None], phi_obs[:, j][:, None],
                          flag_obs[:, j][:, None]], 1)
        out[:, j] = (net(u0[ri], cond) - u[ri, j]).pow(2).mean(
            dim=(1, 2)).sqrt().cpu().numpy()
    return out


@torch.no_grad()
def err_tau(net, u, u0, par, tau_q, te):
    n_tau = tau_q.shape[1]
    out = np.zeros((len(te), n_tau))
    ri = np.asarray(te)
    for j in range(n_tau):
        tq = torch.tensor(tau_q[:, j], dtype=torch.float32, device=u.device)
        cond = torch.cat([par[ri], tq[:, None]], 1)
        out[:, j] = (net(u0[ri], cond) - u[ri, j]).pow(2).mean(
            dim=(1, 2)).sqrt().cpu().numpy()
    return out


def state_to_time(d, tr, te):
    """Can the observed phase be inverted into a time?

    Fits tau = g(phi, flag, params) by nearest-neighbour regression in the
    training set and reports the implied error in t_sing when the observation is
    taken at a known physical time. This is a lower-effort check than a trained
    inverse model: if even this recovers t_sing competitively, the state
    coordinate carries the timing information the scalar head is trying to
    learn.
    """
    from scipy.spatial import cKDTree
    taus = scd.tau_grid()
    ph_tr = np.stack([d['phi'][tr].ravel(),
                      d['flag'][tr].ravel() * 10.0], 1)
    tau_tr = np.tile(taus, len(tr))
    tree = cKDTree(ph_tr)
    ph_te = np.stack([d['phi'][te].ravel(), d['flag'][te].ravel() * 10.0], 1)
    _, idx = tree.query(ph_te, k=8)
    tau_hat = tau_tr[idx].mean(1).reshape(len(te), -1)
    tau_true = np.tile(taus, (len(te), 1))
    # at a known physical time t = tau_true * t_sing, an estimate tau_hat
    # implies t_sing_hat = t / tau_hat
    ok = tau_true > 0.1
    rel = np.abs(tau_true[ok] / np.maximum(tau_hat[ok], 1e-6) - 1)
    return {'median_tsing_err_pct': float(100 * np.median(rel)),
            'mean_abs_tau_err': float(np.abs(tau_hat - tau_true)[ok].mean())}


def run(name, seeds, dev, epochs, data_dir):
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

    # train every seed's models first, so the H3 ensemble can reuse them
    trained = []
    for s in seeds:
        head = train_time_head(T, tr, s, dev, epochs, n_fields, n_par)
        with torch.no_grad():
            lt_pred = head(u0[te], par[te]).cpu().numpy() * n.t_sd + n.t_mu
        ratio = 10.0 ** (lt_pred - d['log_tsing'][te])
        tau_dep = np.clip(np.tile(taus_np, (len(te), 1)) / ratio[:, None], 0, 1)
        net_t, _, _ = train_operator('tau', T, tr, te, s, dev, epochs,
                                     n_fields, n_par)
        net_p, _, _ = train_operator('phase', T, tr, te, s, dev, epochs,
                                     n_fields, n_par)
        net_j = train_joint(T, tr, s, dev, epochs, n_fields, n_par)
        trained.append((s, net_t, net_p, net_j, tau_dep))
        print(f"  [{name}] trained seed {s}", flush=True)
    ens = [t[2] for t in trained]

    rows, spreads = [], []
    for s, net_t, net_p, net_j, tau_dep in trained:
        eB = err_tau(net_t, u, u0, par, tau_dep, te)
        phi_te = phi[te]
        flag_te = flag[te]

        for eps, q in ([(e, 0.0) for e in EPS_GRID]
                       + [(0.0, qq) for qq in Q_GRID if qq > 0]):
            eps_n = eps / n.phi_sd          # dex -> normalised phase units
            noise = torch.tensor(
                rng.normal(0.0, eps_n, size=tuple(phi_te.shape)),
                dtype=torch.float32, device=dev)
            flips = torch.tensor(
                (rng.random(tuple(flag_te.shape)) < q).astype(np.float32),
                device=dev)
            phi_o = phi_te + noise
            flag_o = torch.abs(flag_te - flips)

            eC = err_state([net_p], u, u0, par, phi_o, flag_o, te)
            eH1 = err_joint(net_j, u, u0, par, tau_dep, phi_o, flag_o, te)
            eH2, sp = err_state([net_p], u, u0, par, phi_o, flag_o, te,
                                marginal=(max(eps_n, 1e-6), q),
                                return_spread=True)
            eH3 = err_state(ens, u, u0, par, phi_o, flag_o, te)
            rows.append({'seed': s, 'eps_dex': eps, 'q': q,
                         'B_tau_deployed': float(eB.mean()),
                         'C_state_point': float(eC.mean()),
                         'H1_joint': float(eH1.mean()),
                         'H2_marginalised': float(eH2.mean()),
                         'H3_ensemble': float(eH3.mean()),
                         'B_late': float(eB[:, late].mean()),
                         'C_late': float(eC[:, late].mean()),
                         'H1_late': float(eH1[:, late].mean()),
                         'H2_late': float(eH2[:, late].mean()),
                         'H3_late': float(eH3[:, late].mean())})
            if eps > 0 and q == 0:
                # is the marginal spread an honest error bar?
                spreads.append({'eps_dex': eps, 'seed': s,
                                'mean_spread': float(sp.mean()),
                                'mean_error': float(eH2.mean()),
                                'ratio': float(sp.mean() / eH2.mean())})
            print(f"  [{name} s{s}] eps={eps:.2f} q={q:.2f} | B {eB.mean():.4f}"
                  f"  C {eC.mean():.4f}  H1 {eH1.mean():.4f}"
                  f"  H2 {eH2.mean():.4f}  H3 {eH3.mean():.4f}", flush=True)

    # aggregate over seeds
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
            'eps_grid_dex': EPS_GRID, 'q_grid': Q_GRID,
            'gh_nodes': GH_NODES, 'rows': rows, 'aggregate': agg,
            'marginal_spread_vs_error': spreads,
            'state_to_time': state_to_time(d, tr, te)}


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
    names = list(SYSTEMS) if args.all else [args.system]
    for name in names:
        print(f"=== hybrid {name} ===", flush=True)
        res = run(name, range(args.seeds), dev, args.epochs, args.data)
        json.dump(res, open(os.path.join(OUT, f'hybrid_{name}.json'), 'w'),
                  indent=1)


if __name__ == '__main__':
    main()
