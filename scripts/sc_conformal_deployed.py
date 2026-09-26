#!/usr/bin/env python3
"""E35 (V5 Major Comment 2) -- calibrate on the input path the application uses.

THE DEFECT
----------
`sc_conformal.py` calibrates and tests the emulator on the ORACLE tau grid: it
calls `batch_inputs(prof, par, taus, ...)` with the stored `tau_grid()`, so
every residual it quantiles was produced by a query that already knew t_c.

The application does not know t_c. `e9_inference.py` predicts log t_c from the
members' own scalar head and forms

    tau_hat = t_obs / 10 ** mean_m(log t_c,m),

so a deployed query carries the clock head's error on top of the operator's.
A band calibrated on the oracle path is not a band for the deployed path, and
the manuscript claimed the guarantee for a workflow it had not calibrated.

WHAT THIS DOES
--------------
Repeats the split-conformal construction unchanged -- same five frozen members,
same frozen train/calibration/test split, same score, same alpha -- on BOTH
paths, so the difference is the input path and nothing else:

  oracle    conditioning tau taken from the stored grid  (what sc_conformal did)
  deployed  conditioning tau_hat = tau * t_c / t_c_hat, with t_c_hat the
            ensemble-mean prediction of the members' scalar head, clipped to
            [0, 1] by the same endpoint rule the operator was trained under

The target is unchanged in both: the truth at the stored grid tau. That is the
deployment question -- the user asks for the field at a physical time and gets
whatever the predicted clock maps it to.

Also reported, because the referee asked for each by name:
  * absolute band half-widths in log-field units, as quantiles, not just coverage
  * elementwise coverage OF THE SIMULTANEOUS BAND (same band, both readings)
  * a per-trajectory record: score, half-width, realised error, coverage
    indicator and the gate decision at every rung of the published ladder

PRE-REGISTERED READING, written before the run: if deployed-path coverage falls
below the finite-sample band, the applied guarantee is restated as an
oracle-path guarantee and the deployed number is reported as the measurement it
is. No threshold is re-tuned to rescue it.

GPU only, single-threaded on the host: F9 owns the CPU.
Output: results/statecond/conformal_deployed.json (+ .npz per-trajectory record)
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

torch.set_num_threads(1)

from gravop import dataset as ds                                  # noqa: E402
from gravop import conformal as C                                 # noqa: E402
from train import load_data, split, Norm, make_tensors            # noqa: E402
from sc_conformal import (MEMBERS, load_member, wilson,           # noqa: E402
                          trajectory_scores)

# the absolute error ladder of results/statecond/gated_deployment.json,
# fixed on the in-distribution test split and not re-derived here
LADDER_RMS = [0.004186156438663602, 0.007865383871831,
              0.012717971578240405, 0.015490492247045039,
              0.03861032888293266]


@torch.no_grad()
def predict(nets, tensors, taus, idx, device, tau_q=None, bs=4096):
    """Ensemble mean, spread and truth over every (run, tau) pair in `idx`.

    `tau_q` is (len(idx), n_tau): the conditioning tau actually supplied. The
    TARGET is always the truth at the stored grid tau, whatever was supplied.
    """
    lrho, lv, prof, par, tc, nc = tensors
    n_tau = taus.shape[0]
    pr = np.repeat(idx, n_tau)
    pt = np.tile(np.arange(n_tau), len(idx))
    pos = {int(x): k for k, x in enumerate(idx)}
    mus, sds, tgts = [], [], []
    for s in range(0, len(pr), bs):
        ri, ti = pr[s:s + bs], pt[s:s + bs]
        if tau_q is None:
            q = taus[ti]
        else:
            q = torch.tensor(tau_q[[pos[int(x)] for x in ri], ti],
                             dtype=torch.float32, device=device)
        cond = torch.cat([par[ri], q[:, None]], dim=1)
        stack = torch.stack([net(prof[ri], cond)[0] for net in nets], dim=0)
        mus.append(stack.mean(0).cpu())
        sds.append(stack.std(0, unbiased=True).cpu())
        tgts.append(torch.stack([lrho[ri, ti], lv[ri, ti]], dim=1).cpu())
    return (torch.cat(mus).numpy(), torch.cat(sds).numpy(),
            torch.cat(tgts).numpy())


@torch.no_grad()
def clock_ratio(nets, tensors, idx, device):
    """t_c_true / t_c_hat per run, from the members' own scalar head.

    This is the deployment path of `e9_inference.py:predict`: the head is read
    at a dummy tau (it does not consume tau), the members are averaged in log
    space, and the result is un-normalised with the training split's scale --
    which is the step that must not be forgotten.
    """
    lrho, lv, prof, par, tc, nc = tensors
    ri = np.asarray(idx)
    dummy = torch.zeros(len(ri), 1, device=device)
    cond = torch.cat([par[ri], dummy], dim=1)
    ltc_hat = torch.stack([net(prof[ri], cond)[1][:, 0] for net in nets],
                          0).mean(0)
    return torch.pow(10.0, (tc[ri] - ltc_hat) * NORM_TC_SD).cpu().numpy()


NORM_TC_SD = None          # filled in main(); the un-normalisation scale


def band_report(resid, sigma, q, n_tau, rho_sd, v_sd):
    """Coverage and ABSOLUTE width for one band, read both ways.

    Coverage alone does not say whether a band is useful, so the half-widths
    are returned in log-field units -- the units a user reads -- as quantiles
    over every element, and separately as the per-trajectory maximum, which is
    the width the gate compares against a budget.
    """
    half = q * (sigma + C.EPS)                      # (P, 2, N), normalised
    phys = half * np.array([rho_sd, v_sd])[None, :, None]
    inside = resid <= half
    per_traj = inside.reshape(-1, n_tau, *inside.shape[1:])
    whole = per_traj.reshape(per_traj.shape[0], -1).all(axis=1)
    w = phys.reshape(-1, n_tau, *phys.shape[1:])
    wmax = w.reshape(w.shape[0], -1).max(axis=1)
    return {
        'coverage_elementwise': float(inside.mean()),
        'coverage_whole_trajectory': float(whole.mean()),
        'n_trajectories': int(whole.size),
        'coverage_whole_ci95': wilson(int(whole.sum()), whole.size),
        'halfwidth_dex': {f'p{p}': float(np.percentile(phys, p))
                          for p in (5, 25, 50, 75, 95)},
        'halfwidth_dex_trajectory_max': {
            f'p{p}': float(np.percentile(wmax, p)) for p in (5, 50, 95)},
        'halfwidth_dex_trajectory_max_density': {
            f'p{p}': float(np.percentile(
                w[:, :, 0].reshape(w.shape[0], -1).max(axis=1), p))
            for p in (5, 50, 95)},
        'halfwidth_dex_trajectory_max_velocity': {
            f'p{p}': float(np.percentile(
                w[:, :, 1].reshape(w.shape[0], -1).max(axis=1), p))
            for p in (5, 50, 95)},
        # The two channels carry different physical scales -- rho_sd and v_sd
        # differ by an order of magnitude -- so a quantile pooled over both is
        # a mixture of two distributions and reads as neither. Each channel is
        # therefore reported on its own axis, and the normalised width is
        # reported alongside, because that is the unit the budget ladder of
        # `gated_deployment.json` is fixed in.
        'halfwidth_dex_density': {f'p{p}': float(np.percentile(phys[:, 0], p))
                                  for p in (5, 25, 50, 75, 95)},
        'halfwidth_dex_velocity': {f'p{p}': float(np.percentile(phys[:, 1], p))
                                   for p in (5, 25, 50, 75, 95)},
        'halfwidth_normalised': {f'p{p}': float(np.percentile(half, p))
                                 for p in (5, 25, 50, 75, 95)},
        'channel_scales_dex_per_normalised_unit': {
            'density': float(rho_sd), 'velocity': float(v_sd)},
    }


def main():
    global NORM_TC_SD
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'd1',
                                                   'train.h5'))
    args = ap.parse_args()
    device = torch.device(args.device)
    levels = [0.10, 0.05]
    t0 = time.time()

    d = load_data(args.data)
    tr, va, te = split(len(d['lrho']))
    norm = Norm(d, tr)
    NORM_TC_SD = float(norm.tc_sd)
    tensors = make_tensors(d, norm, device)
    taus_np = ds.tau_grid()
    taus = torch.tensor(taus_np, dtype=torch.float32, device=device)
    n_tau = len(taus_np)
    nets = [load_member(m, device)[0] for m in MEMBERS]
    print(f'{len(nets)} members; split {len(tr)}/{len(va)}/{len(te)}; '
          f'tc_sd {NORM_TC_SD:.4f}')

    paths = {}
    for tag in ('oracle', 'deployed'):
        if tag == 'oracle':
            q_v = q_t = None
            clk = None
        else:
            rv = clock_ratio(nets, tensors, va, device)
            rt = clock_ratio(nets, tensors, te, device)
            q_v = np.clip(taus_np[None, :] * rv[:, None], 0.0, 1.0)
            q_t = np.clip(taus_np[None, :] * rt[:, None], 0.0, 1.0)
            clk = {
                'median_abs_clock_err_pct_calibration':
                    float(100 * np.median(np.abs(1.0 / rv - 1.0))),
                'median_abs_clock_err_pct_test':
                    float(100 * np.median(np.abs(1.0 / rt - 1.0))),
                'clipped_fraction_test': float((taus_np[None, :]
                                                * rt[:, None] > 1.0).mean()),
            }
        mu_v, sd_v, y_v = predict(nets, tensors, taus, va, device, q_v)
        mu_t, sd_t, y_t = predict(nets, tensors, taus, te, device, q_t)
        r_v, r_t = np.abs(y_v - mu_v), np.abs(y_t - mu_t)

        tz_v = trajectory_scores(r_v, sd_v, n_tau)
        tz_t = trajectory_scores(r_t, sd_t, n_tau)
        rec = {'clock': clk, 'n_calibration_trajectories': int(tz_v.size),
               'levels': {}}
        for a in levels:
            q = C.conformal_quantile(tz_v, a)
            rep = band_report(r_t, sd_t, q, n_tau,
                              float(norm.rho_sd), float(norm.v_sd))
            rep['q'] = float(q)
            rep['finite_sample_band'] = list(C.finite_sample_band(tz_v.size, a))
            rep['ties_in_calibration_scores'] = int(
                tz_v.size - np.unique(tz_v).size)
            rec['levels'][f'{1 - a:.2f}'] = rep
        rec['_scores'] = (tz_v, tz_t, r_t, sd_t)
        paths[tag] = rec
        c = rec['levels']['0.90']
        print(f"  {tag:9s} q {c['q']:7.3f}  whole-traj {c['coverage_whole_trajectory']:.4f}"
              f"  elementwise {c['coverage_elementwise']:.4f}"
              f"  median half-width {c['halfwidth_dex']['p50']:.4f} dex")

    # ---- the per-trajectory record the referee asked for, on the deployed path
    tz_v, tz_t, r_t, sd_t = paths['deployed'].pop('_scores')
    paths['oracle'].pop('_scores')
    a = 0.10
    q = C.conformal_quantile(tz_v, a)
    scale = np.array([float(norm.rho_sd), float(norm.v_sd)])[None, :, None]
    half_n = q * (sd_t + C.EPS)                       # normalised field units
    err_n = r_t                                       # normalised field units
    half = half_n * scale                             # dex, per channel
    err = err_n * scale
    def _rms(a, sel=slice(None)):
        b = a.reshape(-1, n_tau, *a.shape[1:])[:, :, sel]
        return np.sqrt((b.reshape(b.shape[0], -1) ** 2).mean(axis=1))
    # The budget ladder of `gated_deployment.json` is fixed on residuals that
    # were never rescaled by the channel standard deviations, so a serving
    # decision is taken in normalised units. The per-channel physical widths
    # are carried alongside, each on its own axis, because the two channels
    # convert to dex through scales that differ by an order of magnitude.
    gate_stat = _rms(half_n)
    realised = _rms(err_n)
    gate_stat_rho, gate_stat_v = _rms(half, 0), _rms(half, 1)
    realised_rho, realised_v = _rms(err, 0), _rms(err, 1)
    covered = tz_t <= q
    decisions = {f'{b:.6f}': (gate_stat <= b) for b in LADDER_RMS}
    np.savez(os.path.join(ROOT, 'results', 'statecond',
                          'conformal_deployed_records.npz'),
             test_index=np.asarray(te), score=tz_t, q=np.full(len(te), q),
             gate_statistic_rms=gate_stat, realised_rms=realised,
             gate_statistic_rms_dex_density=gate_stat_rho,
             gate_statistic_rms_dex_velocity=gate_stat_v,
             realised_rms_dex_density=realised_rho,
             realised_rms_dex_velocity=realised_v,
             covered=covered,
             **{f'served_at_{k}': v for k, v in decisions.items()})

    out = {
        'protocol': 'docs/CONFORMAL_PREREGISTRATION.md',
        'question': 'does the guarantee calibrated on the oracle tau grid hold '
                    'on the input path the application actually uses?',
        'members': MEMBERS,
        'split': {'train': len(tr), 'calibration': len(va), 'test': len(te)},
        'score': 'max over 48 x 128 x 2 of |y - mu| / (sigma + eps), one per '
                 'trajectory',
        'gate_statistic': 'RMS over the trajectory field of the conformal '
                          'half-width, in normalised field units -- the units '
                          'the budget ladder of gated_deployment.json is fixed '
                          'in, so the two are directly comparable; the ladder '
                          'is taken from there and is not re-derived here. '
                          'Per-channel widths in dex are recorded separately, '
                          'one axis each, in the .npz record.',
        'checkpoints': [f'results/runs/{m}/model.pt' for m in MEMBERS],
        'paths': paths,
        'records': 'results/statecond/conformal_deployed_records.npz',
        'wall_seconds': time.time() - t0,
    }
    for k in ('0.90', '0.95'):
        o = paths['oracle']['levels'][k]
        p = paths['deployed']['levels'][k]
        out.setdefault('comparison', {})[k] = {
            'q_oracle': o['q'], 'q_deployed': p['q'],
            'q_ratio': p['q'] / o['q'],
            'coverage_oracle': o['coverage_whole_trajectory'],
            'coverage_deployed': p['coverage_whole_trajectory'],
            'deployed_inside_finite_sample_band':
                bool(p['coverage_whole_trajectory']
                     >= p['finite_sample_band'][0]),
            'halfwidth_ratio_p50': (p['halfwidth_dex']['p50']
                                    / o['halfwidth_dex']['p50']),
        }
    json.dump(out, open(os.path.join(ROOT, 'results', 'statecond',
                                     'conformal_deployed.json'), 'w'), indent=1)
    print('\ncomparison:')
    for k, v in out['comparison'].items():
        print(f"  nominal {k}: q {v['q_oracle']:.3f} -> {v['q_deployed']:.3f} "
              f"({v['q_ratio']:.2f}x), coverage {v['coverage_oracle']:.4f} -> "
              f"{v['coverage_deployed']:.4f}, inside band "
              f"{v['deployed_inside_finite_sample_band']}")
    print('wrote conformal_deployed.json')


if __name__ == '__main__':
    main()
