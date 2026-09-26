"""P13 -- the deployment contract for arm C, honoured or refuted.

WHY THIS EXISTS
---------------
Arm C is reported as "needs no t_sing", and that is true. It is not the whole
contract. Its coordinate is

    phi(t) = log10( A(t) / A(0) ),   b(t) = 1 if t is past argmin A

and BOTH terms hide a deployment requirement the paper has never stated:

  * `A(0)` is the observable at the INITIAL epoch. A single observed snapshot
    does not contain it. Every number in this paper so far assumes it exactly.
  * `b` is computed from the WHOLE trajectory (`data.phase_variables` compares
    against `argmin` over all stored times). It is not a causal quantity: a
    deployed observer at time t cannot evaluate it from the field at time t
    without either having watched since t = 0 or inferring it.

and a third that only appears when you try to measure phi rather than read it
out of the dataset:

  * phi is defined from the observable on the SOLVER grid, but a deployed
    observer sees a field at the STORED resolution. For a sharpening peak those
    are not the same number.

This script measures all three, plus a finite aperture, against the reversal
thresholds E21 established (the phase-measurement error at which arm C stops
beating the best time-conditioned arm: 0.014 dex `burgers`, 0.042 `film`,
0.082 `ks`, 0.138 `rd`, >0.2 `sidm`).

WHAT IS TRAINED
---------------
Only the branch classifier -- a deliberately small CNN on a single stored field,
reported as an achievable rate rather than a bound. Arms A/B/C/D are restored
from checkpoints, so the degradations are evaluated on the published weights.

Output: results/statecond/contract.json
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
from sc_experiment import (LATE, Norm, field_err, load_checkpoint,  # noqa: E402
                           tensors)
from sc_endpoint import check_plain_arm                          # noqa: E402
from sc_timehead import converged_armB                        # noqa: E402
from sc_provenance import stamp                              # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
CKPT = os.path.join(OUT, 'ckpt')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']

# E21/E23: phase-measurement error at which arm C's advantage reverses
REVERSAL_DEX = {'burgers': 0.014, 'film': 0.042, 'ks': 0.082, 'rd': 0.138,
                'sidm': 0.2}


def log_a_from_field(name, u):
    """log10 A(t) as a deployed observer would compute it: from the FIELD.

    The dataset's own `a` is the observable on the solver grid, evaluated
    inside the integrator. Nobody deploying a surrogate has that. What they
    have is a field at the resolution the surrogate consumes, which is what
    this reads. The gap between the two is measured, not assumed away.
    """
    f = u[:, :, 0, :]
    if name in ('rd', 'ks'):        # stored log10 of the field; A = max field
        return f.max(axis=2)
    if name == 'film':              # stored log10 h; A = 1/min h
        return -f.min(axis=2)
    if name == 'sidm':              # stored log10 rho; A = rho_c, innermost
        return f[:, :, 0]
    if name == 'burgers':           # stored u; A = max|du/dx|
        dx = 1.0 / f.shape[-1]
        g = (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dx)
        return np.log10(np.maximum(np.abs(g).max(axis=2), 1e-300))
    raise KeyError(name)


def aperture_log_a(name, u, frac):
    """log10 A measured through a finite aperture: the outer `frac` of the
    domain is unobserved. For the radial systems that is the inner region --
    exactly where the singular structure lives -- which is the honest version
    of the constraint `src/gravop/phase.py` already enforces for the haloes."""
    f = u[:, :, 0, :]
    n = f.shape[-1]
    k = max(1, int(frac * n))
    if name == 'sidm':              # innermost k shells unresolved
        return f[:, :, k]
    if name == 'ks':                # innermost k radial points unresolved
        return f[:, :, k:].max(axis=2)
    if name == 'rd':
        return f[:, :, k:-k].max(axis=2) if 2 * k < n else f.max(axis=2)
    if name == 'film':
        return -f[:, :, k:-k].min(axis=2) if 2 * k < n else -f.min(axis=2)
    if name == 'burgers':
        dx = 1.0 / n
        g = (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dx)
        return np.log10(np.maximum(np.abs(g[:, :, k:-k]).max(axis=2), 1e-300))
    raise KeyError(name)


class BranchNet(nn.Module):
    """Is this field past the observable's minimum? One snapshot, no history.

    Deliberately small: the question is whether the branch is *inferable* from
    a single field at all, so a modest model that succeeds settles it, and one
    that fails only bounds the answer from one side. Reported either way.
    """

    def __init__(self, n_fields, n_x):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_fields, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * (n_x // 4), 64),
                                  nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.body(x))[:, 0]


def train_branch(u, flag, tr, te, dev, epochs, seed):
    # Seed BOTH generators. This used to seed only torch, while the batch order
    # came from `np.random.permutation` on the unseeded global numpy RNG, so
    # the branch classifier was the one thing in this repository that did not
    # reproduce run to run -- caught 2026-09-06 when two identical P22 runs
    # returned 94.3% and 94.2% on `rd`. `sc_experiment.train_operator` seeds
    # both and always did; this function was the outlier.
    torch.manual_seed(seed + 4242)
    np.random.seed(seed + 4242)
    n_runs, n_tau, n_fields, n_x = u.shape
    net = BranchNet(n_fields, n_x).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ri = np.repeat(tr, n_tau)
    ti = np.tile(np.arange(n_tau), len(tr))
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(len(ri))
        for s in range(0, len(perm), 512):
            sel = perm[s:s + 512]
            r, t = ri[sel], ti[sel]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(u[r, t]), flag[r, t])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        pred = torch.stack([net(u[te, j]) for j in range(n_tau)], 1)
    return net, (pred > 0).float().cpu().numpy()


def _load_artefact(fname):
    p = os.path.join(OUT, fname)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--system', default=None)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--backbone', default='fno')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--aperture', type=float, default=0.05)
    ap.add_argument('--sigma-a0', type=float, default=0.05)
    args = ap.parse_args()
    dev = args.device

    path = os.path.join(OUT, 'contract.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    names = SYSTEMS if args.all else [args.system]

    for name in names:
        d = scd.load(os.path.join(DATA, f'{name}.h5'))
        n_runs, n_tau, n_fields, n_x = d['u'].shape
        tr, va, te = scd.split(n_runs)
        n = Norm(d, tr)
        T = tensors(d, n, dev)
        u, u0, par, phi, flag, lts = T
        taus = scd.tau_grid()
        late = taus > LATE

        # ---- the measurement chain, all in dex on phi -------------------
        la_true = np.log10(d['a'])
        phi_true = la_true - la_true[:, :1]
        chains = {}
        la_field = log_a_from_field(name, d['u'])
        chains['resolution'] = la_field - la_field[:, :1]
        la_ap = aperture_log_a(name, d['u'], args.aperture)
        chains['aperture'] = la_ap - la_ap[:, :1]
        rng = np.random.default_rng(20260905)
        # COMMON RANDOM NUMBERS (P22). One standard-normal draw per trajectory,
        # scaled by sigma, rather than an independent draw per sweep node. With
        # independent draws the sweep is not monotone in sigma at 3 seeds x ~60
        # trajectories -- `rd` returned 0.0749 at 0.01 dex and 0.0701 at 0.02 --
        # so the tolerance, which is an interpolated crossing of that curve, was
        # being read off sampling noise. Scaling one draw makes |phi error| grow
        # monotonically per trajectory, so the crossing is well defined, and it
        # makes the sigma = args.sigma_a0 node of the sweep identical to the
        # `full` chain by construction rather than approximately equal.
        z_a0 = rng.normal(0.0, 1.0, size=(n_runs, 1))
        off = args.sigma_a0 * z_a0
        chains['a0_uncertain'] = phi_true - off
        chains['full'] = (la_ap - la_ap[:, :1]) - off
        # A(0) is not a free parameter to guess at: sweep it and report the
        # tolerance at which arm C stops being worth using, which is the
        # contract a deployment has to honour.
        # P22 added 0.005 and 0.03: the crossovers against the corrected
        # comparator land near 0.028 (`rd`) and 0.039 (`sidm`) dex, which the
        # original nodes straddled with a 2.5x gap and could only interpolate.
        sweep_sigmas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
        sweep = {}
        for sig in sweep_sigmas:
            sweep[sig] = (la_ap - la_ap[:, :1]) - sig * z_a0

        # ---- arm C under each chain, from the published weights ---------
        per_seed = {k: [] for k in list(chains) + ['exact', 'branch_pred']}
        sweep_seed = {sig: [] for sig in sweep}
        late_seed = {k: [] for k in per_seed}
        # P23: keep the per-trajectory vector, not just its mean. `field_err`
        # already returns (n_test, n_tau); this script used to average it away
        # at the point of writing, which is why every contract number lacked a
        # trajectory-resampled interval while the superseded E22 numbers had
        # one. Shape per key: (seeds, n_test).
        per_traj = {k: [] for k in per_seed}
        bacc, bacc_turn, bacc_by_rate = [], [], []

        for s in range(args.seeds):
            pp = os.path.join(CKPT, f'{name}_{args.backbone}_phase_seed{s}.pt')
            net_c, mc = load_checkpoint(pp, dev)
            check_plain_arm(mc, pp)

            def cond_with(phi_arr, flag_arr):
                pt = torch.tensor((phi_arr - n.phi_mu) / n.phi_sd,
                                  dtype=torch.float32, device=dev)
                ft = torch.tensor(flag_arr, dtype=torch.float32, device=dev)

                def f(ri, ti):
                    return torch.cat([par[ri], pt[ri, ti][:, None],
                                      ft[ri, ti][:, None]], 1)
                return f

            fl = d['flag']
            e = field_err(net_c, cond_with(phi_true, fl), u, u0, te)
            per_seed['exact'].append(float(e.mean()))
            late_seed['exact'].append(float(e[:, late].mean()))
            per_traj['exact'].append(e.mean(axis=1))
            for k, ph in chains.items():
                e = field_err(net_c, cond_with(ph, fl), u, u0, te)
                per_seed[k].append(float(e.mean()))
                late_seed[k].append(float(e[:, late].mean()))
                per_traj[k].append(e.mean(axis=1))

            for sig, ph in sweep.items():
                e = field_err(net_c, cond_with(ph, fl), u, u0, te)
                sweep_seed[sig].append(float(e.mean()))

            # ---- the branch, inferred rather than supplied ---------------
            bnet, bpred = train_branch(u, flag, tr, te, dev, args.epochs, s)
            truth = fl[te]
            bacc.append(float((bpred == truth).mean()))
            # near the turning point: where |d phi / d tau| is smallest
            dphi = np.abs(np.gradient(phi_true[te], taus, axis=1))
            q = np.quantile(dphi, 0.2)
            m = dphi <= q
            bacc_turn.append(float((bpred[m] == truth[m]).mean()))
            bacc_by_rate.append([
                float((bpred[sel] == truth[sel]).mean()) for sel in
                (dphi <= np.quantile(dphi, 0.25),
                 (dphi > np.quantile(dphi, 0.25)) & (dphi <= np.quantile(dphi, 0.75)),
                 dphi > np.quantile(dphi, 0.75))])

            fl_pred = fl.copy()
            fl_pred[te] = bpred
            e = field_err(net_c, cond_with(chains['full'], fl_pred), u, u0, te)
            per_seed['branch_pred'].append(float(e.mean()))
            late_seed['branch_pred'].append(float(e[:, late].mean()))
            per_traj['branch_pred'].append(e.mean(axis=1))

            del net_c, bnet
            torch.cuda.empty_cache()

        def ms(v):
            return {'mean': float(np.mean(v)), 'std': float(np.std(v))}

        phi_err = {k: float(np.median(np.abs(v[te] - phi_true[te])))
                   for k, v in chains.items()}
        phi_err_p90 = {k: float(np.quantile(np.abs(v[te] - phi_true[te]), 0.9))
                       for k, v in chains.items()}

        rec = {
            'system': name,
            'seeds': args.seeds,
            'aperture_frac': args.aperture,
            'sigma_a0_dex': args.sigma_a0,
            'phi_error_dex_median': phi_err,
            'phi_error_dex_p90': phi_err_p90,
            'reversal_threshold_dex': REVERSAL_DEX[name],
            'nrmse': {k: ms(v) for k, v in per_seed.items()},
            'nrmse_late': {k: ms(v) for k, v in late_seed.items()},
            'branch_accuracy': ms(bacc),
            'branch_accuracy_slow_quintile': ms(bacc_turn),
            'branch_accuracy_by_rate_tercile': np.mean(bacc_by_rate,
                                                       axis=0).tolist(),
            'per_trajectory': {k: np.asarray(v).tolist()
                               for k, v in per_traj.items()},
            'test_index': np.asarray(te).tolist(),
            # the two artefacts whose change must invalidate this one; both
            # have already done so once
            '_provenance': stamp('sc_contract.py', inputs={
                'direct.json': _load_artefact('direct.json'),
                'timehead.json': _load_artefact('timehead.json'),
            }),
        }
        base = rec['nrmse']['exact']['mean']
        rec['cost_vs_exact'] = {k: rec['nrmse'][k]['mean'] / base
                                for k in rec['nrmse']}
        rec['resolution_alone_exceeds_reversal'] = (
            phi_err['resolution'] > REVERSAL_DEX[name])

        # the contract: how well must A(0) be known for arm C to stay ahead of
        # the best arm that needs no measurement at all (arm D, P15)?
        rec['sweep_sigma_a0'] = {str(k): float(np.mean(v))
                                 for k, v in sweep_seed.items()}
        # ---- the comparator, corrected (P22) ----------------------------
        # P13 compared arm C to arm D because arm D was then the best arm that
        # needs no measurement. P20 then made arm B through a CONVERGED clock
        # stronger than arm D in three systems, and this script was never
        # re-run. The comparator is now min(arm B converged, arm D), and the
        # tolerance is quoted against THAT. `arm_D_reference` is retained so
        # the published numbers stay auditable, not because it is the right
        # comparator.
        xs = sorted(sweep_seed)
        ys = [float(np.mean(sweep_seed[x])) for x in xs]

        def tolerance_against(ref):
            """Largest sigma_a0 at which arm C still beats `ref`."""
            if ys[0] >= ref:
                return None                      # loses before any A(0) error
            if ys[-1] < ref:
                return float('inf')              # wins across the whole sweep
            for i in range(1, len(xs)):
                if ys[i - 1] <= ref < ys[i]:
                    f = (ref - ys[i - 1]) / (ys[i] - ys[i - 1])
                    return xs[i - 1] + f * (xs[i] - xs[i - 1])
            return None

        armd = None
        dj = os.path.join(OUT, 'direct.json')
        if os.path.exists(dj):
            dd = json.load(open(dj)).get(name)
            if dd:
                armd = dd['nrmse']['D_time_direct']['mean']
                rec['arm_D_reference'] = armd
                tol_d = tolerance_against(armd)
                rec['a0_tolerance_dex'] = (None if tol_d in (None, float('inf'))
                                           else tol_d)
                rec['arm_C_ever_beats_D'] = ys[0] < armd

        armb, armb_traj = converged_armB(name, T, te, n, d['log_tsing'], dev,
                                         seeds=args.seeds,
                                         backbone=args.backbone)
        if armb is None:
            rec['armB_converged_reference'] = None
            rec['comparator_note'] = (
                'converged clock checkpoints absent: run sc_timehead.py '
                '--all --budgets 20000 --save-heads. The comparator falls '
                'back to arm D, which P20 showed is NOT the strongest time '
                'arm in rd, ks or sidm.')
        else:
            rec['armB_converged_reference'] = armb
            rec['armB_converged_per_trajectory'] = armb_traj.tolist()

        cands = {k: v for k, v in (('D', armd), ('B_converged', armb))
                 if v is not None}
        if cands:
            best_src = min(cands, key=cands.get)
            best = cands[best_src]
            tol_b = tolerance_against(best)
            rec['best_time_reference'] = best
            rec['best_time_source'] = best_src
            rec['a0_tolerance_vs_best_time_dex'] = (
                None if tol_b in (None, float('inf')) else tol_b)
            rec['arm_C_ever_beats_best_time'] = ys[0] < best
            rec['arm_C_full_over_best_time'] = (
                best / rec['nrmse']['full']['mean'])
            # the sweep always carries the aperture and resolution terms, so
            # sigma_a0 = 0 does NOT mean "phi measured perfectly" -- it means
            # "A(0) known exactly, everything else as a real instrument would
            # see it". Say which.
            if tol_b is None:
                rec['contract'] = (
                    f'arm C loses to the best time arm ({best_src}) from the '
                    'aperture and resolution terms alone, before any A(0) '
                    'error')
            elif tol_b == float('inf'):
                rec['contract'] = (
                    f'arm C beats the best time arm ({best_src}) at every '
                    'A(0) tolerance swept')
            else:
                rec['contract'] = (
                    f'A(0) to better than {tol_b:.3f} dex, at the stated '
                    f'aperture and resolution, to beat arm {best_src}')
        out[name] = rec
        print(f"{name:8s} phi err (dex): res {phi_err['resolution']:.4f} "
              f"ap {phi_err['aperture']:.4f} full {phi_err['full']:.4f} "
              f"| reversal {REVERSAL_DEX[name]:.3f} "
              f"| C exact {base:.4f} -> full {rec['nrmse']['full']['mean']:.4f}"
              f" ({rec['cost_vs_exact']['full']:.2f}x) "
              f"| best time {rec.get('best_time_source', '?')} "
              f"{rec.get('best_time_reference', float('nan')):.4f} "
              f"(C/best {rec.get('arm_C_full_over_best_time', float('nan')):.2f}x) "
              f"| branch acc {rec['branch_accuracy']['mean']:.3f} "
              f"(slow {rec['branch_accuracy_slow_quintile']['mean']:.3f})"
              f" | contract: {rec.get('contract', 'n/a')}",
              flush=True)
        json.dump(out, open(path, 'w'), indent=1)

    print('wrote', path)


if __name__ == '__main__':
    main()
