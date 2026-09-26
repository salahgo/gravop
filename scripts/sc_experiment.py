"""Cross-domain test of the singularity-conditioning penalty.

This repeats, on four PDEs that have nothing to do with dark matter, the
measurement that E17/E18/E20 made for gravothermal SIDM haloes. The question is
whether the result is a property of that physical system or a property of the
factorisation used by any surrogate of a finite-time-singularity problem.

The task, stated identically for every system. A user has the initial data and
the physical parameters, and one scalar saying how far the evolution has got.
Predict the field. The arms differ only in what that scalar is and in what it
costs to obtain:

  A  tau-conditioned, oracle
     Scalar = tau = t / t_sing, with the true t_sing supplied. This is the
     number papers usually report, and it is not obtainable in use: t_sing is
     the thing the singularity defines, so it is not known before the fact.

  B  tau-conditioned, deployed
     Scalar = tau computed from a *predicted* t_sing (`TimeHead`, trained on
     the same training split). This is what the same model actually delivers.

  C  state-conditioned
     Scalar = phi = log10(A/A_0) plus the post-minimum flag, read directly off
     the field. No t_sing is involved at any point, so C's measured error is
     already its deployed error.

Backbone, data, splits, seeds, optimiser and budget are identical across arms;
only the conditioning vector changes. The metric is the field RMSE in units of
the training-set field standard deviation (nRMSE), which is defined for signed
and log-transformed fields alike and so is comparable across the four systems.

C is also given a leakage check: phi is measured from the target field itself,
so the error is recomputed on the part of the domain far from where the
observable lives. If C's advantage came from being told a value inside its own
target, it would vanish there.

Usage:  sc_experiment.py --system rd [--seeds 3] [--device cuda:0]
Output: results/statecond/<system>.json
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
import statecond                                          # noqa: E402
from statecond import data as scd                        # noqa: E402
from statecond.model import BACKBONES, TimeHead           # noqa: E402
from statecond.pdes import SYSTEMS                       # noqa: E402

# Import user plugins BEFORE the parser is built: `--backbone` and `--system`
# validate against the registries, so a plugin-registered name has to be
# present by the time argparse constructs its `choices`.
statecond.load_plugins(statecond.plugin_args_from_argv())

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'statecond')
LATE = 0.92          # same late-time cut as the gravothermal experiments
TSING_DELTAS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


class Norm:
    """Field and parameter normalisation from the training split only."""

    def __init__(self, d, tr):
        self.f_mu = d['u'][tr].mean(axis=(0, 1, 3), keepdims=True)
        self.f_sd = d['u'][tr].std(axis=(0, 1, 3), keepdims=True) + 1e-9
        self.p_mu = d['par'][tr].mean(0, keepdims=True)
        self.p_sd = d['par'][tr].std(0, keepdims=True) + 1e-9
        self.t_mu = float(d['log_tsing'][tr].mean())
        self.t_sd = float(d['log_tsing'][tr].std()) + 1e-9
        self.phi_mu = float(d['phi'][tr].mean())
        self.phi_sd = float(d['phi'][tr].std()) + 1e-9


def tensors(d, n, dev):
    u = torch.tensor((d['u'] - n.f_mu) / n.f_sd).float().to(dev)
    u0 = u[:, 0].contiguous()
    par = torch.tensor((d['par'] - n.p_mu) / n.p_sd).float().to(dev)
    phi = torch.tensor((d['phi'] - n.phi_mu) / n.phi_sd).float().to(dev)
    flag = torch.tensor(d['flag']).float().to(dev)
    lts = torch.tensor((d['log_tsing'] - n.t_mu) / n.t_sd).float().to(dev)
    return u, u0, par, phi, flag, lts


def phys_time(d, tr, taus):
    """Standardised log10 PHYSICAL time for every (run, tau) pair.

    The `phys` arm is conditioned on the wall-clock time of the physics, with
    no t_sing factorisation anywhere: t = tau * t_sing at training time, and at
    deployment t is simply the time the user asks about, so unlike arm B there
    is nothing to predict and no oracle/deployed distinction to draw.

    Logs, because t_sing spans up to 3.9 decades across a system's prior and a
    linear scale would collapse every early trajectory onto zero. tau = 0 is
    the one grid point log cannot take, so it is floored at half the first
    non-zero tau -- one point of 48, below which no target exists anyway.
    """
    taus = np.asarray(taus, float)
    floor = 0.5 * taus[taus > 0].min()
    lt = (np.log10(np.maximum(taus, floor))[None, :]
          + np.asarray(d['log_tsing'])[:, None])
    mu, sd = float(lt[tr].mean()), float(lt[tr].std()) + 1e-9
    return ((lt - mu) / sd).astype(np.float32), mu, sd


def train_operator(kind, T, tr, te, seed, dev, epochs, n_fields, n_par,
                   backbone='fno', h1=0.0, extra=None):
    """Train one operator. kind: 'tau', 'phase' or 'phys'.

    `extra` carries the (n_runs, n_tau) conditioning table for 'phys'; the
    other kinds read theirs out of `T`.
    """
    u, u0, par, phi, flag, _ = T
    torch.manual_seed(seed)
    np.random.seed(seed)
    taus = torch.tensor(scd.tau_grid(), dtype=torch.float32, device=dev)
    n_tau = u.shape[1]

    def cond_for(ri, ti):
        if kind == 'tau':
            return torch.cat([par[ri], taus[ti][:, None]], 1)
        if kind == 'phys':
            return torch.cat([par[ri], extra[ri, ti][:, None]], 1)
        return torch.cat([par[ri], phi[ri, ti][:, None],
                          flag[ri, ti][:, None]], 1)

    net = BACKBONES[backbone](n_fields=n_fields, n_par=n_par,
                              cond_extra=1 if kind in ('tau', 'phys') else 2,
                              n_x=u.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ri_all = np.repeat(tr, n_tau)
    ti_all = np.tile(np.arange(n_tau), len(tr))
    t0 = time.time()
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(len(ri_all))
        for s in range(0, len(perm), 1024):
            sel = perm[s:s + 1024]
            ri, ti = ri_all[sel], ti_all[sel]
            pred = net(u0[ri], cond_for(ri, ti))
            tgt = u[ri, ti]
            opt.zero_grad()
            loss = nn.functional.mse_loss(pred, tgt)
            if h1 > 0.0:
                # Empirical H1 term, after Liu et al., J. Comput. Phys. 506
                # (2024) 112944: penalise the first difference as well as the
                # value, which pushes the model to resolve high-frequency
                # structure that the plain L2 loss lets it blur. Here it is an
                # *intervention on spectral bias with the architecture held
                # fixed*, which is what makes it a test of the mechanism in
                # section 5 rather than one more backbone.
                #
                # The term is normalised by the target's own scale ratio, so
                # h1 is dimensionless and h1 = 1 means "balanced against the
                # L2 term". Without this a single constant would be badly
                # mis-scaled across systems: var(field)/var(first difference)
                # ranges from 38 (film) to 2419 (ks) on the production data,
                # so one global weight would do nothing in some systems and
                # dominate in others.
                dp = pred[..., 1:] - pred[..., :-1]
                dt = tgt[..., 1:] - tgt[..., :-1]
                scale = (tgt.var() / dt.var().clamp_min(1e-12)).detach()
                loss = loss + h1 * scale * nn.functional.mse_loss(dp, dt)
            loss.backward()
            opt.step()
        sch.step()
    return net, cond_for, time.time() - t0


def train_time_head(T, tr, seed, dev, epochs, n_fields, n_par):
    u, u0, par, _, _, lts = T
    torch.manual_seed(seed + 777)
    head = TimeHead(n_fields=n_fields, n_par=n_par, n_x=u.shape[-1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_t = torch.tensor(np.asarray(tr), device=dev)
    for _ in range(epochs):
        head.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for s in range(0, len(perm), 256):
            sel = perm[s:s + 256]
            opt.zero_grad()
            nn.functional.mse_loss(head(u0[sel], par[sel]), lts[sel]).backward()
            opt.step()
        sch.step()
    return head


@torch.no_grad()
def field_err(net, cond_fn, u, u0, te, tau_query=None, mask=None):
    """Per-(run, tau) normalised RMSE. If tau_query is given the operator is
    asked at those taus while the target stays at the grid tau."""
    net.eval()
    n_tau = u.shape[1]
    out = np.zeros((len(te), n_tau))
    ri = np.asarray(te)
    for j in range(n_tau):
        ti = np.full(len(te), j)
        cond = cond_fn(ri, ti)
        if tau_query is not None:
            cond = torch.cat([cond[:, :-1],
                              torch.tensor(tau_query[:, j], dtype=torch.float32,
                                           device=u.device)[:, None]], 1)
        diff = net(u0[ri], cond) - u[ri, j]
        if mask is not None:
            diff = diff[:, :, mask]
        out[:, j] = diff.pow(2).mean(dim=(1, 2)).sqrt().cpu().numpy()
    return out


def observable_mask(d, frac=0.15):
    """Grid points far from where the singular observable is measured.

    The observable is an extremum of the field, so its location is taken as the
    argmax of |field - field(t=0)| accumulated over the test trajectories. The
    mask drops a window of `frac` of the domain around it.
    """
    dev_field = np.abs(d['u'][:, -1] - d['u0']).mean(axis=(0, 1))
    c = int(np.argmax(dev_field))
    n = dev_field.size
    half = max(1, int(frac * n / 2))
    keep = np.ones(n, dtype=bool)
    lo, hi = c - half, c + half + 1
    keep[max(0, lo):min(n, hi)] = False
    if lo < 0:
        keep[n + lo:] = False
    if hi > n:
        keep[:hi - n] = False
    return keep, c


def save_checkpoint(path, net, kind, backbone, norm, seed, epochs, shape,
                    h1=0.0):
    """Persist one trained arm.

    Without this, re-evaluating any arm of E22-E27 costs a full retrain, which
    is why several audit questions went unanswered rather than unanswerable.
    The normalisation is stored with the weights because it is fitted on the
    training split and the network is meaningless without it.
    """
    n_fields, n_par, n_x = shape
    torch.save({
        'state_dict': net.state_dict(),
        'kind': kind, 'backbone': backbone, 'seed': seed, 'epochs': epochs,
        'h1': h1,
        'n_fields': n_fields, 'n_par': n_par, 'n_x': n_x,
        'cond_extra': 1 if kind in ('tau', 'phys') else 2,
        'norm': {k: getattr(norm, k) for k in
                 ('f_mu', 'f_sd', 'p_mu', 'p_sd', 't_mu', 't_sd',
                  'phi_mu', 'phi_sd')},
    }, path)


def load_checkpoint(path, dev='cpu'):
    """Rebuild an operator from `save_checkpoint`. Returns (net, meta)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    if ck['kind'] == 'time':
        net = TimeHead(n_fields=ck['n_fields'], n_par=ck['n_par'],
                       n_x=ck['n_x'])
    else:
        net = BACKBONES[ck['backbone']](
            n_fields=ck['n_fields'], n_par=ck['n_par'],
            cond_extra=ck['cond_extra'], n_x=ck['n_x'])
    net.load_state_dict(ck['state_dict'])
    return net.to(dev).eval(), ck


def arm_tag(h1):
    """The suffix that distinguishes an arm trained with the H1 loss term.

    Results and checkpoints must agree on this or they cannot be matched up
    again afterwards; keeping it in one function is what makes that true.
    """
    return '_h1' if h1 > 0.0 else ''


def run_system(name, seeds, dev, epochs, data_dir, backbone='fno', h1=0.0,
               ckpt_dir=None):
    d = scd.load(os.path.join(data_dir, f'{name}.h5'))
    n_runs, n_tau, n_fields, n_x = d['u'].shape
    n_par = d['par'].shape[1]
    tr, va, te = scd.split(n_runs)
    n = Norm(d, tr)
    T = tensors(d, n, dev)
    u, u0, par, phi, flag, lts = T
    taus_np = scd.tau_grid()
    late = taus_np > LATE
    keep, cpos = observable_mask(d)
    keep_t = torch.tensor(keep, device=dev)

    per_seed = {'A_tau_oracle': [], 'B_tau_deployed': [], 'C_state': [],
                'C_state_masked': [], 'A_tau_oracle_masked': [],
                'tsing_err_pct': [], 'fit_seconds_tau': [],
                'fit_seconds_phase': []}
    late_seed = {'A_tau_oracle': [], 'B_tau_deployed': [], 'C_state': []}
    # per-tau error profiles, for the figure showing where the penalty lives
    curves = {'A_tau_oracle': [], 'B_tau_deployed': [], 'C_state': []}
    per_traj = {'A_tau_oracle': [], 'B_tau_deployed': [], 'C_state': []}
    sens = {}

    for s in seeds:
        head = train_time_head(T, tr, s, dev, epochs, n_fields, n_par)
        with torch.no_grad():
            lt_pred = head(u0[te], par[te]).cpu().numpy() * n.t_sd + n.t_mu
        ratio = 10.0 ** (lt_pred - d['log_tsing'][te])       # predicted / true
        per_seed['tsing_err_pct'].append(
            float(100 * np.median(np.abs(ratio - 1))))

        net_t, cond_t, wt = train_operator('tau', T, tr, te, s, dev, epochs,
                                           n_fields, n_par, backbone, h1)
        net_p, cond_p, wp = train_operator('phase', T, tr, te, s, dev, epochs,
                                           n_fields, n_par, backbone, h1)
        per_seed['fit_seconds_tau'].append(wt)
        per_seed['fit_seconds_phase'].append(wp)

        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            shape = (n_fields, n_par, n_x)
            # The tag MUST match the one the results filename gets. It did not
            # until 2026-09-05, so the P10 --h1 run overwrote the P6d plain-FNO
            # checkpoints and every later script that loaded `<sys>_fno_*`
            # silently got the H1 arm (see scripts/sc_ckpt_audit.py). The
            # deployed error is nearly arm-invariant, so nothing downstream
            # noticed; only the ORACLE error tells the arms apart.
            for kind, net in (('tau', net_t), ('phase', net_p),
                              ('time', head)):
                save_checkpoint(
                    os.path.join(ckpt_dir, f'{name}_{backbone}'
                                 f'{arm_tag(h1)}_{kind}_seed{s}.pt'),
                    net, kind, backbone, n, s, epochs, shape, h1)

        eA = field_err(net_t, cond_t, u, u0, te)
        tau_dep = np.clip(np.tile(taus_np, (len(te), 1)) / ratio[:, None], 0, 1)
        eB = field_err(net_t, cond_t, u, u0, te, tau_query=tau_dep)
        eC = field_err(net_p, cond_p, u, u0, te)
        eAm = field_err(net_t, cond_t, u, u0, te, mask=keep_t)
        eCm = field_err(net_p, cond_p, u, u0, te, mask=keep_t)

        # How accurate would the time head have to be? The deployed error is
        # re-measured against imposed t_sing errors rather than the head's own,
        # which answers the objection that the penalty is an artefact of an
        # underfitted head. Symmetrised over sign, since over- and
        # under-predicting t_sing are not equally harmful.
        for delta in TSING_DELTAS:
            acc = []
            for sgn in (+1.0, -1.0):
                tq = np.clip(np.tile(taus_np, (len(te), 1)) / (1 + sgn * delta),
                             0, 1)
                acc.append(field_err(net_t, cond_t, u, u0, te,
                                     tau_query=tq).mean())
            sens.setdefault(delta, []).append(float(np.mean(acc)))

        for k, e in (('A_tau_oracle', eA), ('B_tau_deployed', eB),
                     ('C_state', eC)):
            per_seed[k].append(float(e.mean()))
            late_seed[k].append(float(e[:, late].mean()))
            curves[k].append(e.mean(axis=0).tolist())
            # per-trajectory means, so the test-set sampling error can be
            # bootstrapped later without retraining (ASSESSMENT.md task 5)
            per_traj[k].append(e.mean(axis=1).tolist())
        per_seed['A_tau_oracle_masked'].append(float(eAm.mean()))
        per_seed['C_state_masked'].append(float(eCm.mean()))
        print(f"  [{name} seed {s}] A {eA.mean():.4f}  B {eB.mean():.4f}  "
              f"C {eC.mean():.4f}  (late: {eA[:, late].mean():.4f} / "
              f"{eB[:, late].mean():.4f} / {eC[:, late].mean():.4f})",
              flush=True)

    def agg(v):
        return {'mean': float(np.mean(v)), 'std': float(np.std(v))}

    res = {
        'system': name,
        'n_runs': int(n_runs), 'n_train': int(len(tr)), 'n_test': int(len(te)),
        'n_fields': int(n_fields), 'n_x': int(n_x), 'n_par': int(n_par),
        'seeds': list(seeds), 'epochs': epochs, 'backbone': backbone,
        # sidm enters through sc_import_sidm.py rather than statecond.pdes
        'blowup_factor': (float(SYSTEMS[name].blowup_factor)
                          if name in SYSTEMS else 1e3),
        'tsing_spread_decades': float(
            d['log_tsing'].max() - d['log_tsing'].min()),
        'nrmse': {k: agg(v) for k, v in per_seed.items()
                  if k.startswith(('A_', 'B_', 'C_'))},
        'nrmse_late': {k: agg(v) for k, v in late_seed.items()},
        'tsing_median_err_pct': agg(per_seed['tsing_err_pct']),
        'fit_seconds': {'tau': agg(per_seed['fit_seconds_tau']),
                        'phase': agg(per_seed['fit_seconds_phase'])},
        'observable_index': int(cpos),
        'masked_points': int((~keep).sum()),
        'tau_grid': taus_np.tolist(),
        'curves': {k: np.mean(v, axis=0).tolist() for k, v in curves.items()},
        'per_trajectory': {k: v for k, v in per_traj.items()},
        'test_index': np.asarray(te).tolist(),
        'tsing_sensitivity': {str(k): float(np.mean(v))
                              for k, v in sorted(sens.items())},
    }
    a = np.mean(per_seed['A_tau_oracle'])
    b = np.mean(per_seed['B_tau_deployed'])
    c = np.mean(per_seed['C_state'])
    res['deployment_penalty'] = float(b / a)
    res['state_vs_deployed'] = float(b / c)
    res['state_vs_oracle'] = float(a / c)
    res['state_vs_deployed_late'] = float(
        np.mean(late_seed['B_tau_deployed']) / np.mean(late_seed['C_state']))
    # the t_sing accuracy a time-conditioned model would need to match the
    # state-conditioned one; None if even the smallest imposed error is already
    # worse, or if the whole sweep stays better
    ds = sorted(sens)
    ev = [np.mean(sens[k]) for k in ds]
    req = None
    for i in range(len(ds)):
        if ev[i] >= c:
            if i == 0:
                req = float(ds[0])
            else:
                f = (c - ev[i - 1]) / max(ev[i] - ev[i - 1], 1e-12)
                req = float(ds[i - 1] + f * (ds[i] - ds[i - 1]))
            break
    res['tsing_accuracy_to_match_state'] = req
    res['leak_check'] = {
        'A_masked': agg(per_seed['A_tau_oracle_masked']),
        'C_masked': agg(per_seed['C_state_masked']),
        'state_vs_oracle_masked': float(
            np.mean(per_seed['A_tau_oracle_masked'])
            / np.mean(per_seed['C_state_masked']))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'sc'))
    ap.add_argument('--backbone', default='fno', choices=list(BACKBONES),
                    help='architecture; the conditioning claim should not '
                         'depend on it')
    ap.add_argument('--h1', type=float, default=0.0, metavar='LAMBDA',
                    help='weight on an H1 (first-difference) loss term. '
                         'Non-zero runs the spectral-bias intervention of P10: '
                         'same architecture, same data, same seeds, only the '
                         'loss changes. Results are written with an _h1 '
                         'suffix.')
    ap.add_argument('--plugin', action='append', default=[], metavar='MODULE',
                    help='module or .py file that registers extra systems or '
                         'backbones (see STATECOND_BENCHMARK.md); repeatable, '
                         'and imported before the choices above are built')
    ap.add_argument('--ckpt-dir', nargs='?', default=None,
                    const=os.path.join(OUT, 'ckpt'),
                    help='save every trained arm here (default '
                         'results/statecond/ckpt), so re-evaluating a result '
                         'does not cost a retrain')
    ap.add_argument('--results-dir', default=OUT,
                    help='where the results JSON goes; point it elsewhere to '
                         'regenerate checkpoints without overwriting a '
                         'published result')
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    dev = torch.device(args.device)
    names = list(SYSTEMS) if args.all else [args.system]

    for name in names:
        print(f"=== {name} ===", flush=True)
        res = run_system(name, range(args.seeds), dev, args.epochs, args.data,
                         args.backbone, args.h1, args.ckpt_dir)
        suffix = '' if args.backbone == 'fno' else f'_{args.backbone}'
        suffix += arm_tag(args.h1)
        json.dump(res, open(os.path.join(args.results_dir,
                                         f'{name}{suffix}.json'), 'w'),
                  indent=1)
        print(f"  t_sing median error {res['tsing_median_err_pct']['mean']:.2f}%"
              f" | deployment penalty {res['deployment_penalty']:.2f}x"
              f" | state vs deployed {res['state_vs_deployed']:.2f}x"
              f" (late {res['state_vs_deployed_late']:.2f}x)", flush=True)


if __name__ == '__main__':
    main()
