#!/usr/bin/env python3
"""Coverage events and total fallback for the clock-error propagation study.

Same members, splits and scores as clock_uq.py, in memory. Two coverage events
per trajectory, density channel, supported queries:
  content       at least 90% of the trajectory's supported queries inside the band
                (score: 90th percentile over the trajectory of |error| / s)
  simultaneous  every supported query inside the band
                (score: maximum over the trajectory of |error| / s)
Each band is calibrated by split conformal at nominal 0.90 for its own event.
Two calibration schemes:
  same     sigma_c and the conformal quantile from all 111 calibration haloes
           (as in clock_uq.py; sigma_c is then fitted on the conformal data)
  split    sigma_c from one half of the calibration haloes, the conformal
           quantile from the other half (fixed random partition), so the
           conformal step sees a score fixed in advance
Also reports the fraction of all queries that fall back to the solver when
unsupported queries (predicted phase > 1) are counted. Writes
experiments/research_extensions/clock_uq/clock_uq_events.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]
import clock_uq as U  # noqa: E402

LEVEL = 0.90


def wilson(k, n, z=1.0):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def traj_stat(S, key, b, kind):
    m = S['support'][b]
    if not m.any():
        return None
    r = S['e_dex'][b][m] / np.maximum(S[key][b][m], 1e-6)
    return float(np.percentile(r, 90) if kind == 'content' else r.max())


def quantile(vals):
    v = np.sort(np.array([x for x in vals if x is not None]))
    k = int(np.ceil((len(v) + 1) * LEVEL)) - 1
    return float(v[min(k, len(v) - 1)])


def scores(S, sigma_c):
    S['s_prop'] = np.sqrt(S['s_base'] ** 2 + (S['dmu'] * sigma_c) ** 2)


def evaluate(cal, test, sigma_c, cal_ids):
    for S in (cal, test):
        scores(S, sigma_c)
    out = {}
    for key in ('s_base', 's_prop'):
        out[key] = {}
        for kind in ('content', 'simultaneous'):
            q = quantile([traj_stat(cal, key, b, kind) for b in cal_ids])
            hits = [traj_stat(test, key, b, kind) for b in range(len(test['c_true']))]
            hits = [h for h in hits if h is not None]
            k = int(sum(h <= q for h in hits))
            sup = np.broadcast_to(test['support'][:, :, None], test['mu'].shape)
            out[key][kind] = dict(multiplier=q, covered=k, n=len(hits), coverage=k / len(hits),
                                  coverage_68=wilson(k, len(hits)),
                                  median_halfwidth_dex=float(np.median((q * test[key])[sup])))
    return out


def main():
    import train_baryon_emulator as T
    from gravop import dataset as ds
    d = T.load()
    tr, va, te, ho = T.splits(d)
    q = ds.tau_grid()
    members = [U.load_member(os.path.join(ROOT, f'results/baryons_20261107/runs/seed{s}/model.pt')) for s in range(3)]
    S = {k: U.run_split(d, ids, members, q) for k, ids in (('val', va), ('test', te), ('held', ho))}
    for s in S.values():
        e = np.abs(s['mu'] - s['truth']); e[~s['support']] = np.nan
        s['e_dex'] = e
    v = S['val']
    lnerr = np.log(v['c_bar'] / v['c_true'])
    res = {'level': LEVEL, 'channel': 'log density only', 'queries': 'supported (predicted phase <= 1)'}
    n = len(lnerr)
    res['same'] = {'sigma_c': float(np.sqrt(np.mean(lnerr ** 2)))}
    for split in ('test', 'held'):
        res['same'][split] = evaluate(v, S[split], res['same']['sigma_c'], range(n))
    perm = np.random.default_rng(20261112).permutation(n)
    fit_ids, conf_ids = perm[: n // 2], perm[n // 2:]
    sig = float(np.sqrt(np.mean(lnerr[fit_ids] ** 2)))
    res['split'] = {'sigma_c': sig, 'n_fit': int(len(fit_ids)), 'n_conformal': int(len(conf_ids))}
    for split in ('test', 'held'):
        res['split'][split] = evaluate(v, S[split], sig, conf_ids)
    # total fallback including unsupported queries, at the recall-90 threshold of clock_uq.json
    rec = json.load(open(os.path.join(EXT, 'clock_uq', 'clock_uq.json')))['splits']
    res['fallback'] = {}
    for split, key in (('test', 'test'), ('held', 'held_out')):
        s = S[split]
        unsup = 1.0 - float(s['support'].mean())
        for k in ('s_base', 's_prop'):
            ab = rec[key]['query_level']['gate_val90'][k]['abstain_frac']
            res['fallback'].setdefault(split, {})[k] = dict(
                unsupported_frac=unsup, abstain_supported=ab, abstain_all=unsup + (1 - unsup) * ab)
    res['clock_rel_err_median'] = {sp: float(np.median(np.abs(S[sp]['c_bar'] / S[sp]['c_true'] - 1)))
                                   for sp in ('val', 'test', 'held')}
    json.dump(res, open(os.path.join(EXT, 'clock_uq', 'clock_uq_events.json'), 'w'), indent=1)
    for sch in ('same', 'split'):
        for sp in ('test', 'held'):
            for k in ('s_base', 's_prop'):
                r = res[sch][sp][k]
                print(sch, sp, k, {kind: (round(r[kind]['coverage'], 3), round(r[kind]['median_halfwidth_dex'], 3))
                                   for kind in ('content', 'simultaneous')})
    print(json.dumps(res['fallback'], indent=0)); print(res['clock_rel_err_median'])


if __name__ == '__main__':
    main()
