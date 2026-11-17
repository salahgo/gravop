#!/usr/bin/env python3
"""Experiment D: uncertainty for the baryonic emulator on its own clock.

The deployed emulator (3 baryon-aware members, results/baryons_20261107/runs)
predicts a collapse clock c and the density at phase tau; a query at physical
time t is served at tau = t / c_bar (c_bar = geometric ensemble-mean clock),
exactly as in the R4 study. The served prediction is NOT changed here; only the
uncertainty score attached to it is, so differences are purely about detecting
bad predictions.

Scores per query (dex):
  s_base   member spread at the common clock (R4 / current practice)
  s_own    member spread with each member at its OWN clock, tau_m = t / c_m
           (joint time-and-state ensemble)
  s_prop   sqrt(s_base^2 + (d mu / d ln tau * sigma_c)^2): first-order
           propagation of a clock error, sigma_c calibrated on the validation
           haloes (RMS of ln(c_bar / c_true)); no true clock at inference
  s_both   sqrt(s_own^2 + (d mu / d ln tau * sigma_res)^2), sigma_res the part
           of the validation clock error not explained by member clock spread
Calibration on the validation split only (halo-level split of
train_baryon_emulator.splits), evaluation on test (in-distribution) and on the
held-out deepest potentials (out of distribution: no guarantee claimed).
Outputs experiments/research_extensions/clock_uq/{clock_uq.json, clock_uq.npz}.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUT = os.path.join(EXT, 'clock_uq')
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BAD = 0.10          # trajectory is bad if its mean relative density error exceeds 10% (R4)
TARGET_RECALL = 0.9


def load_member(path):
    from gravop.model import GravOpNet
    from types import SimpleNamespace
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck.get('args', {})
    m = GravOpNet(n_r=128, width=a.get('width', 64), modes=a.get('modes', 32),
                  n_layers=a.get('layers', 4), cond_dim=a.get('cond_dim', 6))
    m.load_state_dict(ck['model'])
    return m.to(DEV).eval(), SimpleNamespace(**ck['norm'])


@torch.no_grad()
def predict(m, n, prof, par, tau):
    """prof (B,2,128), par (B,P), tau (B,Q) -> log rho (B,Q,128) at clipped tau."""
    B, Q = tau.shape
    out = np.empty((B, Q, 128), np.float32)
    for b in range(B):
        cond = torch.cat([par[b:b + 1].expand(Q, -1),
                          torch.tensor(np.clip(tau[b], 0, 1), dtype=torch.float32, device=DEV)[:, None]], 1)
        f, _ = m(prof[b:b + 1].expand(Q, -1, -1), cond)
        out[b] = (f[:, 0] * n.rho_sd + n.rho_mu).cpu().numpy()
    return out


@torch.no_grad()
def clocks(m, n, prof, par):
    _, hh = m(prof, torch.cat([par, torch.zeros(len(par), 1, device=DEV)], 1))
    return 10 ** (hh[:, 0].cpu().numpy() * n.tc_sd + n.tc_mu)


def auroc(score, bad):
    s, y = np.asarray(score), np.asarray(bad, bool)
    if y.all() or (~y).all():
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def run_split(d, ids, members, q):
    import train_baryon_emulator as T
    n0 = members[0][1]
    lrho_t, lv_t, prof, par, tc, _ = T.tensors(d, n0, DEV, baryon_inputs=True)
    prof, par = prof[ids], par[ids]
    c_m = np.array([clocks(m, n, prof, par) for m, n in members])          # (M, B)
    c_bar = np.exp(np.log(c_m).mean(0))
    c_true = 10 ** d['log_tc'][ids]
    t_phys = q[None, :] * c_true[:, None]                                  # query times
    tau_bar = t_phys / c_bar[:, None]
    support = (tau_bar >= 0) & (tau_bar <= 1)
    P_bar = np.array([predict(m, n, prof, par, tau_bar) for m, n in members])        # (M,B,Q,R)
    P_own = np.array([predict(m, n, prof, par, t_phys / c_m[k][:, None]) for k, (m, n) in enumerate(members)])
    eps = 0.02
    mu = P_bar.mean(0)
    mu_p = np.array([predict(m, n, prof, par, tau_bar * np.exp(eps)) for m, n in members]).mean(0)
    mu_m = np.array([predict(m, n, prof, par, tau_bar * np.exp(-eps)) for m, n in members]).mean(0)
    dmu = (mu_p - mu_m) / (2 * eps)                                        # d log rho / d ln tau
    truth = d['lrho'][ids][:, :, :]                                        # (B,Q,R) on q grid
    return dict(c_m=c_m, c_bar=c_bar, c_true=c_true, support=support, mu=mu, dmu=dmu, tau_bar=tau_bar,
                s_base=P_bar.std(0), s_own=P_own.std(0), truth=truth,
                held_out=d['held_out'][ids])


def main():
    import train_baryon_emulator as T
    from gravop import dataset as ds
    t0 = time.time()
    d = T.load()
    tr, va, te, ho = T.splits(d)
    q = ds.tau_grid()
    members = [load_member(os.path.join(ROOT, f'results/baryons_20261107/runs/seed{s}/model.pt')) for s in range(3)]
    S = {k: run_split(d, ids, members, q) for k, ids in (('val', va), ('test', te), ('held', ho))}
    # calibration (validation haloes only)
    v = S['val']
    lnerr = np.log(v['c_bar'] / v['c_true'])
    sigma_c = float(np.sqrt(np.mean(lnerr ** 2)))
    lnspread = np.log(v['c_m']).std(0)
    sigma_res = float(np.sqrt(max(np.mean(lnerr ** 2) - np.mean(lnspread ** 2), 0.0)))
    for s in S.values():
        s['s_prop'] = np.sqrt(s['s_base'] ** 2 + (s['dmu'] * sigma_c) ** 2)
        s['s_both'] = np.sqrt(s['s_own'] ** 2 + (s['dmu'] * sigma_res) ** 2)
        e_dex = np.abs(s['mu'] - s['truth'])
        e_dex[~s['support']] = np.nan
        s['e_dex'] = e_dex
        er = np.abs(10 ** (s['mu'] - s['truth']) - 1)          # R4's definition
        er[~s['support']] = np.nan
        s['e_rel'] = er
        # naive baseline score: closeness to the predicted collapse (phase only)
        s['s_phase'] = np.broadcast_to((s['tau_bar'])[:, :, None], s['mu'].shape).copy()
    scores = ('s_base', 's_own', 's_prop', 's_both')

    def traj(s, key):
        x = s[key].copy()
        x[~s['support']] = np.nan
        return np.nanmean(x, axis=(1, 2))

    res = dict(sigma_c_ln=sigma_c, sigma_res_ln=sigma_res,
               val_clock_rel_median=float(np.median(np.abs(v['c_bar'] / v['c_true'] - 1))), splits={})
    # thresholds chosen on validation for TARGET_RECALL of bad trajectories (any unsupported query
    # also raises the gate, as in R4); the R4 fixed 0.02 dex threshold is reported for s_base
    thr = {}
    for k in scores:
        sv, ev = traj(v, k), np.nanmean(v['e_rel'], axis=(1, 2))
        badv = ev > BAD
        unsup = ~v['support'].all(1)
        cand = np.sort(np.unique(sv[badv & ~unsup])) if (badv & ~unsup).any() else np.array([np.inf])
        # largest threshold whose recall on validation is >= target
        best = cand[0]
        for c in cand:
            rec = np.mean((sv[badv] >= c) | unsup[badv])
            if rec >= TARGET_RECALL:
                best = c
        thr[k] = float(best)
    thr_fixed = {'s_base@0.02': 0.02}
    # split-conformal multiplier (trajectory level): r = 90th percentile over a trajectory's
    # supported queries of |error| / score; quantile over validation trajectories
    conf = {}
    for k in scores:
        r = []
        for b in range(len(v['c_true'])):
            m = v['support'][b]
            if m.any():
                r.append(np.nanpercentile((v['e_dex'][b][m] / np.maximum(v[k][b][m], 1e-6)).ravel(), 90))
        r = np.sort(np.array(r))
        nv = len(r)
        kq = int(np.ceil((nv + 1) * 0.9)) - 1
        conf[k] = float(r[min(kq, nv - 1)])
    for name, s in (('test', S['test']), ('held_out', S['held'])):
        e = np.nanmean(s['e_rel'], axis=(1, 2))
        bad = e > BAD
        unsup = ~s['support'].all(1)
        out = dict(n=int(len(e)), bad=int(bad.sum()), unsupported_traj=int(unsup.sum()),
                   median_traj_err=float(np.nanmedian(e)), methods={})
        for k in list(scores) + ['s_base@0.02']:
            kk = k.split('@')[0]
            sc = traj(s, kk)
            th = thr_fixed.get(k, thr.get(k))
            gate = (sc >= th) | unsup
            tp, fn = int((gate & bad).sum()), int((~gate & bad).sum())
            fp = int((gate & ~bad).sum())
            rec = dict(threshold=th, recall=tp / max(bad.sum(), 1), missed=fn, false_alarms=fp,
                       false_alarm_rate=fp / max((~bad).sum(), 1), precision=tp / max(tp + fp, 1),
                       abstain_frac=float(gate.mean()), auroc=auroc(np.where(unsup, np.inf, sc), bad),
                       spearman=float(__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(sc, e).correlation))
            if '@' not in k:
                inside, cover_traj = [], []
                for b in range(len(e)):
                    m = s['support'][b]
                    if m.any():
                        ok = s['e_dex'][b][m] <= conf[kk] * s[kk][b][m]
                        inside.append(ok.mean())
                        cover_traj.append(ok.mean() >= 0.9)
                rec.update(conformal_multiplier=conf[kk], query_coverage=float(np.mean(inside)),
                           traj_coverage_90=float(np.mean(cover_traj)),
                           median_halfwidth_dex=float(np.nanmedian(np.where(s['support'][:, :, None], conf[kk] * s[kk], np.nan))))
            out['methods'][k] = rec
        # query-level detection of bad queries (relative error > 10%) among supported queries
        sup = np.broadcast_to(s['support'][:, :, None], s['mu'].shape)
        badq = (s['e_rel'] > BAD)[sup]
        out['query_level'] = dict(n=int(sup.sum()), bad_frac=float(badq.mean()),
                                  auroc={k: auroc(s[k][sup], badq) for k in list(scores) + ['s_phase']})
        # validation-calibrated query gate (90% recall of bad validation queries) and a
        # trajectory-clustered bootstrap of the AUROC gain over the current spread
        supv = np.broadcast_to(v['support'][:, :, None], v['mu'].shape)
        badv_q = (v['e_rel'] > BAD)[supv]
        gates = {}
        for k in scores:
            th = float(np.quantile(v[k][supv][badv_q], 1 - TARGET_RECALL))
            g = s[k][sup] >= th
            gates[k] = dict(threshold=th, recall=float(g[badq].mean()), abstain_frac=float(g.mean()),
                            precision=float(badq[g].mean()) if g.any() else None,
                            served_bad_frac=float(badq[~g].mean()) if (~g).any() else None)
        out['query_level']['gate_val90'] = gates
        rng = np.random.default_rng(0)
        B = len(s['c_true'])
        traj_id = np.broadcast_to(np.arange(B)[:, None, None], s['mu'].shape)[sup]
        order = np.argsort(traj_id, kind='stable')
        starts = np.searchsorted(traj_id[order], np.arange(B))
        ends = np.searchsorted(traj_id[order], np.arange(B), side='right')
        sb, sp_, yb = s['s_base'][sup][order], s['s_prop'][sup][order], badq[order]
        diffs = []
        for _ in range(200):
            pick = rng.integers(0, B, B)
            idx = np.concatenate([np.arange(starts[i], ends[i]) for i in pick])
            a1, a0 = auroc(sp_[idx], yb[idx]), auroc(sb[idx], yb[idx])
            if a1 is not None and a0 is not None:
                diffs.append(a1 - a0)
        out['query_level']['auroc_gain_prop_vs_base_ci95'] = [float(np.percentile(diffs, 2.5)),
                                                                float(np.percentile(diffs, 97.5))]
        res['splits'][name] = out
    rev = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    res['meta'] = dict(git=rev, device=DEV, wall_s=time.time() - t0, n_val=int(len(va)), n_test=int(len(te)),
                       n_held=int(len(ho)), bad_definition='mean relative density error > 10% over supported queries',
                       recall_target_on_val=TARGET_RECALL, command=' '.join(sys.argv))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'clock_uq.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(OUT, 'clock_uq.npz'),
                        **{f'{sp}_{k}': S[sp][k] for sp in S for k in ('c_m', 'c_bar', 'c_true', 'held_out')})
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
