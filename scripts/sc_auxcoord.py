"""P24 -- is the win from the SINGULAR observable, or from any state functional?

WHY THIS EXISTS
---------------
Arm C conditions on `phi = log10(A(t)/A(0))`, where `A` is the observable that
diverges. Every result in this project compares that against arms conditioned
on a *time* coordinate. Nothing compares it against another arm conditioned on
the state, so two very different claims are currently indistinguishable:

  (a) "condition on a functional of the current state rather than on time"  --
      general, and older than this project: it is the progress-variable idea of
      Pierce & Moin (JFM 504, 2004), transferred to a new setting; or
  (b) "condition on the SINGULAR observable specifically"                   --
      specific, and it would make the choice of `A` part of the method rather
      than an unexamined convention.

A referee will ask which. This script answers it by building arm E, conditioned
on a state functional that carries no privileged relationship to the
singularity:

    psi(t) = log10( ||u(t)||_2 / ||u(0)||_2 )

a bulk norm contrast. It is a functional of exactly the same field arm C reads,
measured at the same instant, standardised the same way, and it goes into the
same conditioning slot -- so the ONLY thing that differs from arm C is which
functional of the state the operator is told.

PRE-REGISTERED, as a fork, because both branches are informative:
  * arm E within 1.5x of arm C in >= 4 systems  -> the claim is (a). More
    general, more robust, and it obliges an explicit positioning against the
    progress-variable literature.
  * arm E >= 3x worse than arm C in >= 4 systems -> the claim is (b). Sharper,
    and the observable-selection rule becomes part of the method.
  * anything else is a split, and is reported as a split.

ALSO HERE (P24b) -- the information-equivalence check
-----------------------------------------------------
Arm C beats the ORACLE arm A in all five systems, which reads like leakage.
It is not, and the structural reason is cheap to establish: `phi` is a
deterministic function of `(u0, theta, tau)`, so arm C carries no more
information than arm A carries. If that holds, arm C beating arm A is a
statement about *learnability* -- one coordinate is better conditioned than the
other -- and not about one arm being told more. `--equiv` fits `phi` from
(u0, theta, tau) with the same head class used for `t_sing` and reports the
held-out residual as a fraction of `phi`'s range.

Usage:
  sc_auxcoord.py --all [--seeds 3] [--device cuda:0]
  sc_auxcoord.py --all --equiv
Output: results/statecond/auxcoord.json
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

import statecond.data as scd                                     # noqa: E402
from statecond.model import TimeHead                             # noqa: E402
from sc_experiment import (LATE, Norm, field_err, load_checkpoint,  # noqa: E402
                           save_checkpoint, tensors, train_operator)
from sc_endpoint import check_plain_arm                          # noqa: E402
from sc_provenance import stamp                                  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(OUT, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']


def psi_table(d, tr):
    """Standardised log10 of the bulk L2 contrast, per (run, tau).

    Computed on the STORED field in its raw units, exactly as a deployed
    observer would compute it, then standardised on the training split like
    every other conditioning channel. `u` is (R, T, C, N); the norm is taken
    over channels and space, so for the multi-field system (`sidm`) it is a
    single scalar summarising the whole state rather than one channel of it.

    A floor is applied before the log because `film` stores log10 h, which can
    make the raw L2 norm small but never zero; the floor only guards against a
    degenerate trajectory rather than acting on any real one.
    """
    u = np.asarray(d['u'], dtype=np.float64)
    nrm = np.sqrt((u ** 2).mean(axis=(2, 3)))            # (R, T)
    psi = np.log10(np.maximum(nrm, 1e-300)
                   / np.maximum(nrm[:, :1], 1e-300))
    mu, sd = float(psi[tr].mean()), float(psi[tr].std()) + 1e-9
    return ((psi - mu) / sd).astype(np.float32), psi, mu, sd


def equivalence_residual(d, T, tr, te, n, dev, epochs=400, seed=0):
    """Fit phi from (u0, theta, tau); return the held-out residual.

    Same head class as the t_sing predictor, with tau appended to the parameter
    vector. If phi really is a deterministic function of (u0, theta, tau) then
    this residual is near zero and arm C's advantage over arm A cannot be an
    information advantage.
    """
    u, u0, par, phi, flag, _ = T
    n_runs, n_tau = phi.shape
    taus = torch.tensor(scd.tau_grid(), dtype=torch.float32, device=dev)
    torch.manual_seed(seed + 991)
    head = TimeHead(n_fields=u.shape[2], n_par=par.shape[1] + 1,
                    n_x=u.shape[-1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ri_all = np.repeat(tr, n_tau)
    ti_all = np.tile(np.arange(n_tau), len(tr))
    for _ in range(epochs):
        head.train()
        perm = np.random.permutation(len(ri_all))
        for s in range(0, len(perm), 1024):
            sel = perm[s:s + 1024]
            ri, ti = ri_all[sel], ti_all[sel]
            pin = torch.cat([par[ri], taus[ti][:, None]], 1)
            opt.zero_grad()
            nn.functional.mse_loss(head(u0[ri], pin), phi[ri, ti]).backward()
            opt.step()
        sch.step()
    head.eval()
    res = []
    with torch.no_grad():
        for j in range(n_tau):
            ri = np.asarray(te)
            pin = torch.cat([par[ri], taus[j].repeat(len(ri))[:, None]], 1)
            res.append((head(u0[ri], pin) - phi[ri, j]).cpu().numpy())
    res = np.stack(res, 1)                                # (n_te, n_tau)
    # report in the ORIGINAL phi units (dex), and as a fraction of the range
    rng = float(np.ptp(np.asarray(d['phi'])))
    rmse_dex = float(np.sqrt((res ** 2).mean()) * n.phi_sd)
    return {'rmse_dex': rmse_dex, 'phi_range_dex': rng,
            'frac_of_range': rmse_dex / rng if rng else None,
            'epochs': epochs}


def run_system(name, seeds, dev, epochs, backbone, do_equiv, save):
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    n_runs, n_tau, n_fields, n_x = d['u'].shape
    n_par = d['par'].shape[1]
    tr, va, te = scd.split(n_runs)
    n = Norm(d, tr)
    T = tensors(d, n, dev)
    u, u0, par, phi, flag, _ = T
    taus = scd.tau_grid()
    late = taus > LATE

    ptab, psi_raw, psi_mu, psi_sd = psi_table(d, tr)
    ptab_t = torch.tensor(ptab, device=dev)

    acc = {k: [] for k in ('A_tau_oracle', 'C_state', 'E_aux_state')}
    late_acc = {k: [] for k in acc}
    per_traj = {k: [] for k in acc}

    for s in range(seeds):
        # arm E: the only thing trained here. Same call signature as arm D,
        # because 'phys' is simply "condition on this scalar table" -- the
        # table is what differs.
        net_e, cond_e, _ = train_operator('phys', T, tr, te, s, dev, epochs,
                                          n_fields, n_par, backbone,
                                          extra=ptab_t)
        eE = field_err(net_e, cond_e, u, u0, te)
        if save:
            os.makedirs(CKPT, exist_ok=True)
            save_checkpoint(os.path.join(
                CKPT, f'{name}_{backbone}_aux_seed{s}.pt'),
                net_e, 'phys', backbone, n, s, epochs, (n_fields, n_par, n_x))

        ptau = os.path.join(CKPT, f'{name}_{backbone}_tau_seed{s}.pt')
        ppha = os.path.join(CKPT, f'{name}_{backbone}_phase_seed{s}.pt')
        net_a, ma = load_checkpoint(ptau, dev)
        net_c, mc = load_checkpoint(ppha, dev)
        for meta, path in ((ma, ptau), (mc, ppha)):
            check_plain_arm(meta, path)
        taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)

        def cond_a(ri, ti):
            return torch.cat([par[ri], taus_t[ti][:, None]], 1)

        def cond_c(ri, ti):
            return torch.cat([par[ri], phi[ri, ti][:, None],
                              flag[ri, ti][:, None]], 1)

        eA = field_err(net_a, cond_a, u, u0, te)
        eC = field_err(net_c, cond_c, u, u0, te)
        for k, e in (('A_tau_oracle', eA), ('C_state', eC),
                     ('E_aux_state', eE)):
            acc[k].append(float(e.mean()))
            late_acc[k].append(float(e[:, late].mean()))
            per_traj[k].append(e.mean(axis=1))
        del net_e, net_a, net_c
        torch.cuda.empty_cache()

    def ms(v):
        return {'mean': float(np.mean(v)), 'std': float(np.std(v))}

    out = {
        'system': name, 'backbone': backbone, 'seeds': seeds,
        'epochs': epochs, 'n_test': len(te),
        'psi_definition': 'log10( ||u(t)||_2 / ||u(0)||_2 ), bulk L2 contrast',
        'psi_range_dex': float(np.ptp(psi_raw)),
        'phi_range_dex': float(np.ptp(np.asarray(d['phi']))),
        'nrmse': {k: ms(v) for k, v in acc.items()},
        'nrmse_late': {k: ms(v) for k, v in late_acc.items()},
        'per_trajectory': {k: np.asarray(v).tolist()
                           for k, v in per_traj.items()},
        'test_index': np.asarray(te).tolist(),
    }
    C, E, A = (out['nrmse']['C_state']['mean'],
               out['nrmse']['E_aux_state']['mean'],
               out['nrmse']['A_tau_oracle']['mean'])
    out['E_over_C'] = E / C
    out['E_over_A'] = E / A
    out['aux_within_1p5x_of_singular'] = bool(E / C <= 1.5)
    out['aux_at_least_3x_worse'] = bool(E / C >= 3.0)
    if do_equiv:
        out['phi_from_tau_residual'] = equivalence_residual(
            d, T, tr, te, n, dev)
    out['_provenance'] = stamp('sc_auxcoord.py')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--backbone', default='fno')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--equiv', action='store_true',
                    help='also run P24b, the phi-from-(u0,theta,tau) fit')
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()
    dev = torch.device(args.device)

    path = os.path.join(OUT, 'auxcoord.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    names = SYSTEMS if args.all else [args.system]

    for name in names:
        r = run_system(name, args.seeds, dev, args.epochs, args.backbone,
                       args.equiv, not args.no_save)
        out[name] = r
        eq = r.get('phi_from_tau_residual')
        print(f"{name:8s} A {r['nrmse']['A_tau_oracle']['mean']:.4f}  "
              f"C {r['nrmse']['C_state']['mean']:.4f}  "
              f"E {r['nrmse']['E_aux_state']['mean']:.4f}  "
              f"| E/C {r['E_over_C']:.2f}x  E/A {r['E_over_A']:.2f}x"
              + (f"  | phi|tau residual {eq['frac_of_range']*100:.2f}% "
                 f"of range" if eq else ''), flush=True)
        json.dump(out, open(path, 'w'), indent=1)

    done = [k for k in SYSTEMS if k in out]
    if len(done) == len(SYSTEMS):
        ratios = [out[k]['E_over_C'] for k in SYSTEMS]
        near = [k for k in SYSTEMS if out[k]['aux_within_1p5x_of_singular']]
        far = [k for k in SYSTEMS if out[k]['aux_at_least_3x_worse']]
        if len(near) >= 4:
            verdict = ('(a) ANY state functional: the bulk norm is within 1.5x '
                       'of the singular observable in >= 4 systems, so the '
                       'claim generalises and must be positioned against the '
                       'progress-variable literature')
        elif len(far) >= 4:
            verdict = ('(b) the SINGULAR observable specifically: the bulk '
                       'norm is >= 3x worse in >= 4 systems, so observable '
                       'selection is part of the method')
        else:
            verdict = ('SPLIT: neither pre-registered branch is met; report as '
                       'a split rather than choosing one')
        out['_summary'] = {
            'E_over_C_range': [min(ratios), max(ratios)],
            'within_1p5x': near, 'at_least_3x_worse': far,
            'verdict': verdict,
        }
        json.dump(out, open(path, 'w'), indent=1)
        print('\n' + verdict)
    print('wrote', path)


if __name__ == '__main__':
    main()
