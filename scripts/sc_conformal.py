#!/usr/bin/env python3
"""P37 — calibrated deployment guarantees for the emulator.

Protocol and fixed consequences: docs/CONFORMAL_PREREGISTRATION.md.

Split conformal over the published 5-member ensemble. Nothing is retrained;
the weights are loaded read-only and calibration is a quantile of held-out
residuals. Calibrates on the frozen validation split, evaluates on the frozen
test split, then applies the SAME threshold to the two out-of-distribution sets
to measure how far the guarantee travels.

GPU only, and deliberately single-threaded on the host: F9 owns the CPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

torch.set_num_threads(1)                      # do not compete with F9

from gravop.model import GravOpNet            # noqa: E402
from gravop import dataset as ds              # noqa: E402
from gravop import conformal as C             # noqa: E402
from train import load_data, split, Norm, make_tensors, batch_inputs  # noqa: E402

MEMBERS = [f'gravop_s{i}' for i in range(5)]


def wilson(k, n, z=1.96):
    """Empirical coverage on a finite test set has binomial scatter; the
    conformal band is a statement about EXPECTED coverage, so comparing a
    point estimate to it without an interval would be a category error."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def load_member(run, device):
    meta = json.load(open(os.path.join(ROOT, 'results', 'runs', run,
                                       'metrics.json')))
    a = meta['args']
    net = GravOpNet(n_r=ds.N_R, width=a['width'], modes=a['modes'],
                    n_layers=a['layers']).to(device)
    sd = torch.load(os.path.join(ROOT, 'results', 'runs', run, 'model.pt'),
                    map_location=device)
    net.load_state_dict(sd['model'] if 'model' in sd else sd)
    net.eval()
    return net, meta


@torch.no_grad()
def ensemble_predict(nets, tensors, taus, idx, device, bs=4096):
    """Per-member fields over every (run, tau) pair in `idx`.

    Returns (mean, spread, truth) each (P, 2, N). Spread is the unbiased
    standard deviation across members, which is the quantity the project has
    been using as an uncertainty and the one being audited here.
    """
    lrho, lv, prof, par, tc, nc = tensors
    n_tau = taus.shape[0]
    pr = np.repeat(idx, n_tau)
    pt = np.tile(np.arange(n_tau), len(idx))
    mus, sds, tgts = [], [], []
    for s in range(0, len(pr), bs):
        ri, ti = pr[s:s + bs], pt[s:s + bs]
        pi, ci = batch_inputs(prof, par, taus, ri, ti)
        stack = torch.stack([net(pi, ci)[0] for net in nets], dim=0)
        mus.append(stack.mean(0).cpu())
        sds.append(stack.std(0, unbiased=True).cpu())
        tgts.append(torch.stack([lrho[ri, ti], lv[ri, ti]], dim=1).cpu())
    return (torch.cat(mus).numpy(), torch.cat(sds).numpy(),
            torch.cat(tgts).numpy())


def trajectory_scores(resid, sigma, n_tau):
    """One score per trajectory: the worst normalised residual anywhere in it.

    The elementwise treatment above is not a valid conformal procedure here, and
    the reason is NOT that the elements are correlated -- correlation does not
    preclude exchangeability, as identical copies of one variable show.
    Exchangeability is required of the SCORES, and the 48 x 128 x 2 elements of
    a trajectory are indexed by snapshot, radius, channel and parent
    trajectory: the joint law is not invariant under permutations that move an
    element across those indices. Trajectories, drawn i.i.d. from the prior and
    split at random, are. Scoring each trajectory by its maximum turns the
    guarantee into the one a user actually wants: with probability 1-alpha the
    WHOLE predicted field of a new trajectory lies inside its band.
    """
    r = resid.reshape(-1, n_tau, *resid.shape[1:])
    g = sigma.reshape(-1, n_tau, *sigma.shape[1:])
    z = r / (g + C.EPS)
    return z.reshape(z.shape[0], -1).max(axis=1)


def report(name, resid, sigma, cals, nominal_levels):
    out = {'n_elements': int(resid.size)}
    for a in nominal_levels:
        nom = 1.0 - a
        cov_c = float(C.covers(resid, sigma, cals[a]).mean())
        cov_p = float(C.covers(resid, sigma, cals[('plain', a)]).mean())
        cov_g, z = C.gaussian_coverage(resid, sigma, nom)
        out[f'{nom:.2f}'] = {
            'conformal_normalised': cov_c,
            'conformal_plain': cov_p,
            'gaussian_ensemble': cov_g,
            'gaussian_z': z,
            'median_halfwidth_normalised':
                float(np.median(C.interval_width(sigma, cals[a]))),
            'halfwidth_plain': float(cals[('plain', a)]['q']),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'd1',
                                                   'train.h5'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'statecond',
                                                  'conformal.json'))
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = [0.10, 0.05]

    d = load_data(args.data)
    tr, va, te = split(len(d['lrho']))
    norm = Norm(d, tr)
    tensors = make_tensors(d, norm, device)
    taus = torch.tensor(ds.tau_grid(), dtype=torch.float32, device=device)

    nets = [load_member(m, device)[0] for m in MEMBERS]
    print(f'{len(nets)} members; split {len(tr)}/{len(va)}/{len(te)}')

    t0 = time.time()
    mu_v, sd_v, y_v = ensemble_predict(nets, tensors, taus, va, device)
    mu_t, sd_t, y_t = ensemble_predict(nets, tensors, taus, te, device)
    r_v, r_t = np.abs(y_v - mu_v), np.abs(y_t - mu_t)

    cals = {}
    for a in levels:
        cals[a] = C.calibrate(r_v, sd_v, a, normalise=True)
        cals[('plain', a)] = C.calibrate(r_v, sd_v, a, normalise=False)

    # the valid procedure: one exchangeable score per trajectory
    n_tau = taus.shape[0]
    tz_v = trajectory_scores(r_v, sd_v, n_tau)
    tz_t = trajectory_scores(r_t, sd_t, n_tau)
    traj = {}
    for a in levels:
        q = C.conformal_quantile(tz_v, a)
        traj[f'{1 - a:.2f}'] = {
            'q': q,
            'n_calibration_trajectories': int(tz_v.size),
            'finite_sample_band': list(C.finite_sample_band(tz_v.size, a)),
            'coverage_test': float(np.mean(tz_t <= q)),
            'coverage_test_ci95': wilson(int(np.sum(tz_t <= q)), tz_t.size),
        }

    out = {
        'protocol': 'docs/CONFORMAL_PREREGISTRATION.md',
        'members': MEMBERS,
        'split': {'train': len(tr), 'calibration': len(va), 'test': len(te)},
        'n_calibration_elements': int(r_v.size),
        'finite_sample_band': {f'{1 - a:.2f}':
                               list(C.finite_sample_band(r_v.size, a))
                               for a in levels},
        'thresholds': {f'{1 - a:.2f}': {'q_normalised': cals[a]['q'],
                                        'q_plain': cals[('plain', a)]['q']}
                       for a in levels},
        'in_distribution_test': report('test', r_t, sd_t, cals, levels),
        'trajectory_level': traj,
        'note_exchangeability': (
            'Elementwise coverage is reported for comparison with the '
            'neural-operator conformal literature. It is not valid here -- not '
            'because the elements are correlated (correlation does not '
            'preclude exchangeability) but because they are indexed by '
            'snapshot, radius, channel and parent trajectory, so the joint law '
            'of the scores is not permutation-invariant across those indices. '
            'Trajectories are drawn i.i.d. and split at random, so they are '
            'exchangeable; trajectory_level carries the valid guarantee, the '
            'whole field of a new trajectory inside its band.'),
    }

    # the same threshold, carried outside the training families
    for tag, path in (('ood_ic', 'ood_ic.h5'), ('ood_param', 'ood_param.h5')):
        p = os.path.join(ROOT, 'data', 'd1', path)
        if not os.path.exists(p):
            continue
        dd = load_data(p)
        tt = make_tensors(dd, norm, device)
        allidx = np.arange(len(dd['lrho']))
        mu_o, sd_o, y_o = ensemble_predict(nets, tt, taus, allidx, device)
        r_o = np.abs(y_o - mu_o)
        out[tag] = report(tag, r_o, sd_o, cals, levels)
        tz_o = trajectory_scores(r_o, sd_o, n_tau)
        out[tag]['trajectory_level'] = {
            k: {'coverage': float(np.mean(tz_o <= traj[k]['q'])),
                'n_trajectories': int(tz_o.size)} for k in traj}
        out[tag]['mean_spread_vs_test'] = float(sd_o.mean() / sd_t.mean())
        out[tag]['mean_residual_vs_test'] = float(r_o.mean() / r_t.mean())

    # ---- gating economics (arXiv:2608.23075), from measured timings
    bench = json.load(open(os.path.join(ROOT, 'results', 'e8', 'bench.json')))
    solver_s = float(np.median([b['wall_s'] for b in bench['solver']]))
    # GPU only: the emulator is a GPU code, so its cost is a GPU cost. Taking
    # the median over every entry at batch >= 64 mixes two CUDA timings with two
    # CPU ones and returns the midpoint between them, which is the cost of no
    # machine that exists and overstates the emulator by 21x.
    emu = [e for e in bench['emulator']
           if e['kind'] == 'trajectory' and e['batch'] >= 64
           and str(e['device']).startswith('cuda')]
    assert emu, 'no GPU emulator timings in bench.json'
    emu_s = float(np.median([e['seconds_per_item'] for e in emu]))
    r = emu_s / solver_s                      # surrogate / solver, per call
    g = len(MEMBERS) * emu_s / solver_s       # the gate is the ensemble itself
    # The gate IS the ensemble and its mean is the served answer, so r is inside
    # g rather than added to it; the reference form would charge it twice.
    d_break = C.break_even_deferral_fused(g)

    # what a calibrated gate would actually defer: the fraction of trajectories
    # whose worst normalised residual exceeds the conformal threshold
    defer = {}
    for a in levels:
        k = f'{1 - a:.2f}'
        q = traj[k]['q']
        defer[k] = {'in_distribution': float(np.mean(tz_t > q))}
        for tag in ('ood_ic', 'ood_param'):
            if tag in out:
                defer[k][tag] = 1.0 - out[tag]['trajectory_level'][k]['coverage']
    out['gating_economics'] = {
        'solver_seconds_per_trajectory': solver_s,
        'emulator_seconds_per_trajectory': emu_s,
        'cost_ratio_r': r,
        'gate_cost_share_g': g,
        'break_even_deferral': d_break,
        'required_deferral': defer,
        'cost_model': 'C_hybrid/C_solver = g + d; g subsumes the surrogate, '
                      'so d_break = 1 - g',
        'reference': 'arXiv:2608.23075 gives (1-g-r)/(1-r) for a gate that is '
                     'a side computation; that is not this workflow',
        'break_even_deferral_reference_formula': C.break_even_deferral(r, g),
    }
    print(f"\ngating economics: r={r:.3e}  g={g:.3e}  "
          f"break-even deferral {d_break:.5f}")
    for k, v in defer.items():
        print(f"  nominal {k}: defer in-dist {v['in_distribution']:.3f}"
              + ''.join(f"  {t} {v[t]:.3f}" for t in ('ood_ic', 'ood_param')
                        if t in v))

    out['wall_seconds'] = time.time() - t0
    json.dump(out, open(args.out, 'w'), indent=1)

    print(f"\n{'set':16s} {'nominal':>8s} {'conformal':>10s} "
          f"{'plain':>8s} {'gaussian':>9s}")
    for tag in ('in_distribution_test', 'ood_ic', 'ood_param'):
        if tag not in out:
            continue
        for a in levels:
            k = f'{1 - a:.2f}'
            b = out[tag][k]
            print(f"{tag:16s} {k:>8s} {b['conformal_normalised']:10.4f} "
                  f"{b['conformal_plain']:8.4f} {b['gaussian_ensemble']:9.4f}")
    print(f"\ntrajectory-level (the valid unit), n_cal="
          f"{tz_v.size}")
    for k, v in traj.items():
        print(f"  nominal {k}: test coverage {v['coverage_test']:.4f}  "
              f"band {v['finite_sample_band'][0]:.4f}-"
              f"{v['finite_sample_band'][1]:.4f}")
        for tag in ('ood_ic', 'ood_param'):
            if tag in out:
                print(f"      {tag:10s} {out[tag]['trajectory_level'][k]['coverage']:.4f}")
    print('\nwrote', os.path.relpath(args.out, ROOT))


if __name__ == '__main__':
    main()
