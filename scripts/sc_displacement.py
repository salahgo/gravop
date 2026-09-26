"""P12 — the clock-error floor, measured on the solver alone.

WHY THIS EXISTS
---------------
Section 5 of the manuscript reports `penalty ~ sensitivity^0.81` and attributes
it to spectral bias: a better operator resolves finer structure, so it is more
sensitive to being asked at the wrong phase. A recomputation over all 25
(system x backbone) points showed that story cannot be what is happening --
within every system the ABSOLUTE error under an imposed t_sing error is nearly
invariant to the backbone (spread 1.01x-1.94x) while the oracle error spans
2.8x-8.5x. Both axes of the "law" are (system constant)/E_A, so the penalty
rises because its denominator falls, not because its numerator rises.

If that reading is right, the constant is not a property of the network at all.
It is a property of the PDE: asking for the solution at the wrong phase costs
you whatever the solution changes by over that phase interval, and no operator,
however good, can do better than that. This script measures exactly that
quantity WITHOUT A NETWORK.

WHAT IS COMPUTED
----------------
For a converged reference trajectory G, the deployed arm asks for the field at
`tau_hat = t / t_sing_hat`. If `t_sing_hat = t_sing (1 + d)` then the operator
is queried at `tau / (1 + d)` while the target remains at `tau`. A PERFECT
operator therefore still returns

    G(tau / (1 + d))   against a target of   G(tau),

and makes an error

    F(d) = || G(tau / (1 + d)) - G(tau) ||          <- the floor.

F is computed here by resampling the stored reference trajectories, in exactly
the normalised-field units (`Norm`, fitted on the training split) and over
exactly the test split and tau grid that `sc_experiment.field_err` uses, so the
numbers are directly comparable to the network's.

THE PRE-REGISTERED PREDICTION
-----------------------------
If the network approximates G with an error independent of the clock offset,

    E(d)^2  ~=  E_A^2 + F(d)^2,

so `sqrt(max(E(d)^2 - E_A^2, 0))` should equal F(d) for every backbone, and F
should account for the backbone-invariance of the numerator. The residual of
that relation is the part of the deployment penalty that IS attributable to the
network -- which is where a spectral-bias account would have to live.

Note that this is the same quadrature step the referee (correctly) objected to
in the N-body error budget. The difference is that here it is the hypothesis
under test and its residual is reported, not an assumption used to license a
subtraction.

Usage:  sc_displacement.py [--out results/statecond/floor.json] [--interp linear|pchip]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from statecond import data as scd                       # noqa: E402
from sc_experiment import Norm, TSING_DELTAS, LATE      # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
SYSTEMS = ('rd', 'ks', 'film', 'burgers', 'sidm')
BACKBONES = ('', '_deeponet', '_cno', '_transformer', '_h1')


def _weights(taus, tq):
    """Linear interpolation index/weight pair for querying `taus` at `tq`."""
    idx = np.clip(np.searchsorted(taus, tq, side='right') - 1,
                  0, len(taus) - 2)
    dt = taus[idx + 1] - taus[idx]
    w = np.clip((tq - taus[idx]) / dt, 0.0, 1.0)
    return idx, w


def resample_tau(un, taus, tq, kind='linear'):
    """Fields at `tq`, interpolated along the tau axis.

    un: (R, T, C, N) normalised fields on the grid `taus`.
    Returns (R, len(tq), C, N).

    Linear is the default because it is the same rule `statecond.data.resample`
    uses to build the dataset, so the floor and the training targets are
    interpolated consistently. `pchip` is offered as a robustness check: if the
    floor is an artefact of coarse interpolation near the singularity the two
    will disagree.
    """
    if kind == 'pchip':
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(taus, un, axis=1, extrapolate=False)(
            np.clip(tq, taus[0], taus[-1]))
    idx, w = _weights(taus, tq)
    wf = w[None, :, None, None]
    return (1 - wf) * un[:, idx] + wf * un[:, idx + 1]


def floor_for_system(name, data_dir, interp='linear'):
    d = scd.load(os.path.join(data_dir, f'{name}.h5'))
    n_runs = d['u'].shape[0]
    tr, _, te = scd.split(n_runs)
    n = Norm(d, tr)
    un = (d['u'] - n.f_mu) / n.f_sd            # the units field_err reports in
    taus = scd.tau_grid()
    late = taus > LATE

    out = {}
    for delta in TSING_DELTAS:
        acc, acc_late = [], []
        # symmetrised over sign, exactly as the sensitivity sweep is
        for sgn in (+1.0, -1.0):
            tq = np.clip(taus / (1.0 + sgn * delta), 0.0, 1.0)
            up = resample_tau(un, taus, tq, interp)
            diff = up[te] - un[te]                       # (R_te, T, C, N)
            e = np.sqrt((diff ** 2).mean(axis=(2, 3)))   # (R_te, T)
            acc.append(float(e.mean()))
            acc_late.append(float(e[:, late].mean()))
        out[str(delta)] = float(np.mean(acc))
        out[f'{delta}_late'] = float(np.mean(acc_late))
    return out, int(len(te))


def floor_deployed(name, data_dir, ckpt_dir, seeds=(0, 1, 2),
                   head_suffix=''):
    """The floor under the time head's OWN per-trajectory error.

    `floor_for_system` imposes a single delta on every trajectory, which matches
    how the sensitivity sweep is defined but not how arm B is deployed: there
    each trajectory carries its own t_sing error, and the map delta -> error is
    concave, so a single median delta does not summarise the ensemble. This
    reruns the floor with the head's actual per-trajectory ratio, which is the
    quantity that should predict E_B.

    The time head does not depend on the backbone (`train_time_head` takes no
    backbone argument), so the fno checkpoints serve for every point.
    """
    import torch
    from sc_experiment import load_checkpoint

    d = scd.load(os.path.join(data_dir, f'{name}.h5'))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    un = (d['u'] - n.f_mu) / n.f_sd
    taus = scd.tau_grid()
    late = taus > LATE
    u0 = torch.tensor(un[:, 0]).float()
    par = torch.tensor((d['par'] - n.p_mu) / n.p_sd).float()
    ri = np.arange(len(te))[:, None]

    acc, acc_late, errs, per_traj = [], [], [], []
    for s in seeds:
        p = os.path.join(ckpt_dir,
                         f'{name}_fno_time{head_suffix}_seed{s}.pt')
        if not os.path.exists(p):
            return None
        head, _ = load_checkpoint(p, 'cpu')
        with torch.no_grad():
            lt = head(u0[te], par[te]).cpu().numpy() * n.t_sd + n.t_mu
        ratio = 10.0 ** (lt - d['log_tsing'][te])        # predicted / true
        errs.append(float(100 * np.median(np.abs(ratio - 1))))
        tq = np.clip(taus[None, :] / ratio[:, None], 0.0, 1.0)
        idx, w = _weights(taus, tq)
        ute = un[te]
        up = ((1 - w)[..., None, None] * ute[ri, idx]
              + w[..., None, None] * ute[ri, idx + 1])
        e = np.sqrt(((up - ute) ** 2).mean(axis=(2, 3)))
        acc.append(float(e.mean()))
        acc_late.append(float(e[:, late].mean()))
        per_traj.append(e.mean(axis=1))                  # (R_te,)
    return {'floor_deployed': float(np.mean(acc)),
            'floor_deployed_late': float(np.mean(acc_late)),
            'tsing_median_err_pct': float(np.mean(errs)),
            'per_trajectory': np.mean(per_traj, axis=0).tolist(),
            'seeds': list(seeds)}


def paired_bootstrap(floor_traj, net_traj, n_boot=10000, seed=20260904):
    """95% CI on mean(net)/mean(floor), resampling TEST TRAJECTORIES.

    Paired: the same trajectory contributes its floor and its network error to
    each resample, so the ratio's uncertainty is not inflated by treating them
    as independent samples. The trajectory is the unit because that is what the
    test split draws, matching `sc_significance.py`.
    """
    f = np.asarray(floor_traj, float)
    g = np.asarray(net_traj, float)
    assert f.shape == g.shape, (f.shape, g.shape)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(f), size=(n_boot, len(f)))
    r = g[idx].mean(axis=1) / f[idx].mean(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def load_points():
    """(system, backbone) -> oracle error and absolute error under imposed d."""
    pts = []
    for s in SYSTEMS:
        for b in BACKBONES:
            p = os.path.join(ROOT, 'results', 'statecond', f'{s}{b}.json')
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            bt = r.get('per_trajectory', {}).get('B_tau_deployed')
            pts.append({
                'system': s,
                'backbone': b.lstrip('_') or 'fno',
                'oracle': r['nrmse']['A_tau_oracle']['mean'],
                'deployed': r['nrmse']['B_tau_deployed']['mean'],
                'deployed_late': r['nrmse_late']['B_tau_deployed']['mean'],
                'tsing_err_pct': r['tsing_median_err_pct']['mean'],
                'imposed': {k: float(v)
                            for k, v in r['tsing_sensitivity'].items()},
                # mean over seeds of the per-trajectory deployed error, so the
                # floor/network ratio can be bootstrapped over trajectories
                'per_trajectory': (np.mean(bt, axis=0).tolist() if bt else None),
            })
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(ROOT, 'data', 'sc'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'statecond',
                                                  'floor.json'))
    ap.add_argument('--interp', default='linear', choices=('linear', 'pchip'))
    ap.add_argument('--ckpt-dir', default=os.path.join(
        ROOT, 'results', 'statecond', 'ckpt'))
    ap.add_argument('--head-suffix', default='',
                    help="'_conv' re-runs the deployed comparison against the "
                         'converged time heads of P20 instead of the '
                         'published undertrained ones')
    a = ap.parse_args()

    floors, n_test, dep = {}, {}, {}
    for s in SYSTEMS:
        floors[s], n_test[s] = floor_for_system(s, a.data_dir, a.interp)
        dep[s] = floor_deployed(s, a.data_dir, a.ckpt_dir,
                                head_suffix=a.head_suffix)
        extra = (f'   deployed {dep[s]["floor_deployed"]:.4f} '
                 f'@ {dep[s]["tsing_median_err_pct"]:.2f}%' if dep[s] else '')
        print(f'{s:9s} floor(10%) = {floors[s]["0.1"]:.4f}   '
              f'late {floors[s]["0.1_late"]:.4f}   (n_test {n_test[s]}){extra}',
              flush=True)

    pts = load_points()
    # the test: does the floor account for the absolute imposed-error numerator?
    rows = []
    for p in pts:
        f = floors[p['system']]
        for dk, ev in p['imposed'].items():
            fl = f[dk]
            resid = ev ** 2 - p['oracle'] ** 2
            net_only = float(np.sqrt(resid)) if resid > 0 else 0.0
            rows.append({
                'system': p['system'], 'backbone': p['backbone'],
                'delta': float(dk),
                'oracle': p['oracle'],
                'imposed_abs': ev,          # E(d), the network's absolute error
                'floor': fl,                # F(d), the solver's, no network
                'quadrature_residual': net_only,   # sqrt(E(d)^2 - E_A^2)
                'ratio_resid_over_floor': net_only / fl if fl > 0 else None,
                'ratio_imposed_over_floor': ev / fl if fl > 0 else None,
            })
    # the late window, where the paper's "last 8%" claims live
    late_rows = [{
        'system': p['system'], 'backbone': p['backbone'],
        'deployed_late': p['deployed_late'],
        'floor_deployed_late': dep[p['system']]['floor_deployed_late'],
        'ratio': p['deployed_late'] / dep[p['system']]['floor_deployed_late'],
    } for p in pts if dep.get(p['system'])]

    at10 = [r for r in rows if r['delta'] == 0.1]
    rr = np.array([r['ratio_resid_over_floor'] for r in at10])
    print(f'\nAt d = 10%, sqrt(E^2 - E_A^2) / F over {len(at10)} points:'
          f'  median {np.median(rr):.2f}  range {rr.min():.2f}-{rr.max():.2f}')

    # The primary statement needs NO quadrature: where the oracle error is small
    # against the floor, E(d) itself should equal F(d).
    strong = [r for r in at10 if r['backbone'] != 'deeponet']
    sv = np.array([r['ratio_imposed_over_floor'] for r in strong])
    print(f'Direct (no quadrature), E(10%)/F over the {len(sv)} points whose '
          f'oracle error is small against the floor:'
          f'  median {np.median(sv):.3f}  range {sv.min():.3f}-{sv.max():.3f}')

    # the deployed test: does the per-trajectory floor predict E_B?
    dep_rows = []
    for p in pts:
        f = dep.get(p['system'])
        if not f:
            continue
        resid = p['deployed'] ** 2 - p['oracle'] ** 2
        net = float(np.sqrt(resid)) if resid > 0 else 0.0
        row = {
            'system': p['system'], 'backbone': p['backbone'],
            'oracle': p['oracle'], 'deployed': p['deployed'],
            'floor_deployed': f['floor_deployed'],
            'ratio_direct': p['deployed'] / f['floor_deployed'],
            'quadrature_residual': net,
            'ratio': net / f['floor_deployed'],
        }
        if p['per_trajectory'] is not None:
            lo, hi = paired_bootstrap(f['per_trajectory'], p['per_trajectory'])
            row['ratio_direct_ci95'] = [lo, hi]
        dep_rows.append(row)
    if dep_rows:
        dr = np.array([r['ratio'] for r in dep_rows])
        print(f'Deployed: sqrt(E_B^2 - E_A^2) / F(head) over {len(dr)} points:'
              f'  median {np.median(dr):.2f}  range {dr.min():.2f}-{dr.max():.2f}')
        print('\nPaired trajectory-level bootstrap, E_B / F(head), 95% CI:')
        for r in dep_rows:
            if 'ratio_direct_ci95' in r:
                lo, hi = r['ratio_direct_ci95']
                print(f"  {r['system']:9s} {r['backbone']:12s} "
                      f"{r['ratio_direct']:.3f}  [{lo:.3f}, {hi:.3f}]")

    lr = np.array([r['ratio'] for r in late_rows])
    print(f"\nLate window (tau > {LATE}), E_B_late / F_late over {len(lr)} "
          f"points:  median {np.median(lr):.2f}  "
          f"range {lr.min():.2f}-{lr.max():.2f}")

    json.dump({
        'floor': floors,
        'floor_deployed': dep,
        'deployed_comparison': dep_rows,
        'late_comparison': late_rows,
        'n_test': n_test,
        'interp': a.interp,
        'deltas': TSING_DELTAS,
        'late_cut': LATE,
        'comparison': rows,
        'note': ('F(d) is the error a PERFECT operator still makes when the '
                 'singular time is wrong by d: it contains no network. '
                 'quadrature_residual is sqrt(E(d)^2 - E_A^2), the part of the '
                 "network's imposed-error response not explained by its oracle "
                 'error; the pre-registered prediction is that it equals F(d).'),
    }, open(a.out, 'w'), indent=1)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
