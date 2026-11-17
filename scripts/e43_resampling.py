"""E43 resampling -- seed-wise scores and exact system-resampling intervals for
the reserved-draw predictor score.

`results/e43/prospective.json` stores the quadrature rule's within-system RMS
log10 residual per system and its mean. The manuscript also quotes the mean
scored one training seed at a time and an interval from resampling the five
systems, which were computed by hand and never stored. This script computes
them from the stored points:

* per seed: the same score with each point's E_A, E_dep and F taken from that
  seed's row instead of the RMS over seeds;
* resampling: five systems drawn with replacement give at most 126 distinct
  multisets, so the bootstrap distribution is enumerated exactly (each
  multiset weighted by its multinomial probability) instead of sampled. The
  interval is the 2.5th-97.5th percentile of that exact distribution, for the
  rule's mean score and for the factor by which it improves on the fitted
  constant (ratio of means).

Output: results/e43/resampling.json
"""
import itertools
import json
import os
from math import factorial

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'e43')
SYSTEMS = ('rd', 'ks', 'film', 'burgers', 'sidm')


def quad_score(points, pick):
    per = {}
    for s in SYSTEMS:
        v = []
        for k, p in points.items():
            if k.split('_', 1)[0] != s:
                continue
            q = pick(p)
            pred = np.hypot(q['E_A'], q['F']) / q['E_A']
            v.append(np.log10(pred / (q['E_dep'] / q['E_A'])))
        per[s] = float(np.sqrt(np.mean(np.square(v))))
    return per, float(np.mean(list(per.values())))


def exact_bootstrap(stat):
    vals, wts = [], []
    for combo in itertools.combinations_with_replacement(range(len(SYSTEMS)),
                                                         len(SYSTEMS)):
        cnt = np.bincount(combo, minlength=len(SYSTEMS))
        wts.append(factorial(len(SYSTEMS)) / np.prod([factorial(c) for c in cnt]))
        vals.append(stat(list(combo)))
    vals, wts = np.array(vals), np.array(wts) / np.sum(wts)
    order = np.argsort(vals)
    cw = np.cumsum(wts[order])

    def q(p):
        return float(vals[order][np.searchsorted(cw, p)])
    return {'n_multisets': int(len(vals)), 'p2.5': q(0.025), 'median': q(0.5),
            'p97.5': q(0.975), 'min': float(vals.min()), 'max': float(vals.max())}


def main():
    d = json.load(open(os.path.join(RES, 'prospective.json')))
    pts = d['points']
    stored = d['rules']['R1_quadrature']
    per, mean = quad_score(pts, lambda p: p)
    assert abs(mean - stored['mean_of_system_rms']) < 1e-12, (mean, stored)

    n_seeds = len(next(iter(pts.values()))['per_seed'])
    seeds = {}
    for s in range(n_seeds):
        _, m = quad_score(pts, lambda p, s=s: p['per_seed'][s])
        seeds[s] = m

    q = np.array([stored['per_system'][s] for s in SYSTEMS])
    c = np.array([d['rules']['R0_constant']['per_system'][s] for s in SYSTEMS])
    out = {
        'source': 'results/e43/prospective.json',
        'quadrature_mean': float(q.mean()),
        'constant_mean': float(c.mean()),
        'factor_over_constant': float(c.mean() / q.mean()),
        'per_seed_mean': seeds,
        'resampling_quadrature_mean': exact_bootstrap(lambda i: q[i].mean()),
        'resampling_factor_over_constant':
            exact_bootstrap(lambda i: c[i].mean() / q[i].mean()),
        'method': ('exact enumeration of the bootstrap over the five systems; '
                   'each multiset weighted by its multinomial probability'),
    }
    json.dump(out, open(os.path.join(RES, 'resampling.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
