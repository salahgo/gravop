#!/usr/bin/env python3
"""Score the coefficient-free rule against a comparator that is allowed to fit.

WHY THIS EXISTS
---------------
The predictor study compared the quadrature rule against a fitted constant and
a fitted power law in the sensitivity exponent, and reported that it wins. Both
of those comparators are weak in the same way: neither is allowed to learn the
relationship between the displacement ratio and the penalty, which is the one
thing a reader would try first. A referee who fits

    log10 p = a + b log10(F / E_A)

leave-one-system-out gets a better mean score than the quadrature rule, and a
paper that has not reported that is claiming an advantage it has not tested
against the obvious alternative.

So it is tested here, on exactly the footing the published comparators use:
fitted on the four retained systems, scored on the fifth, residuals combined as
an RMS within the held-out system and then averaged over the five. The
three-coefficient variant that adds log10(E_A) is scored alongside, because a
comparator that is allowed two coefficients should also be allowed three.

WHAT IT FINDS
-------------
The two-coefficient regression scores better on the mean and is not resolved
against the quadrature rule: it wins in two systems of five, and a bootstrap
over systems puts the difference astride zero. The reading the manuscript takes
from that is the conservative one -- the coefficient-free rule MATCHES a fitted
comparator rather than beating it -- which is a weaker claim than the one the
constant and the power law supported, and a more defensible one.

Every number here is recomputed from `decomposition.json`; nothing is read from
the published predictor scores. That the harness reproduces the published
constant and quadrature scores to the digit is asserted before anything new is
written, so a drift in either artefact fails the run rather than reaching the
table.
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
N_RESAMPLE = 20000
SEED = 20260920


def load():
    d = json.load(open(os.path.join(RES, 'decomposition.json')))
    return [(k.split('_')[0], k.split('_', 1)[1], v)
            for k, v in d.items() if not k.startswith('_')]


def penalty(v):
    return np.log10(v['E_dep'] / v['E_A'])


FEATURES = {
    # name: (n coefficients, feature map)
    'constant': (1, lambda v: [1.0]),
    'ratio_regression': (2, lambda v: [1.0, np.log10(v['F'] / v['E_A'])]),
    'two_feature_regression': (3, lambda v: [1.0,
                                             np.log10(v['F'] / v['E_A']),
                                             np.log10(v['E_A'])]),
}


def loo_fitted(rows, systems, fmap):
    """Fit on the retained systems, score on the held-out one."""
    per = {}
    for s in systems:
        train = [(fmap(v), penalty(v)) for ss, _, v in rows if ss != s]
        A = np.array([a for a, _ in train])
        t = np.array([b for _, b in train])
        beta, *_ = np.linalg.lstsq(A, t, rcond=None)
        per[s] = float(np.sqrt(np.mean(
            [(float(np.dot(fmap(v), beta)) - penalty(v)) ** 2
             for ss, _, v in rows if ss == s])))
    return per


def quadrature(rows, systems):
    """No coefficients, so nothing is held out from it."""
    def pred(v):
        return np.log10(np.sqrt(v['E_A'] ** 2 + v['F'] ** 2) / v['E_A'])
    return {s: float(np.sqrt(np.mean([(pred(v) - penalty(v)) ** 2
                                      for ss, _, v in rows if ss == s])))
            for s in systems}


def main():
    rows = load()
    systems = sorted({s for s, _, _ in rows})
    scores = {'quadrature': quadrature(rows, systems)}
    for name, (_, fmap) in FEATURES.items():
        scores[name] = loo_fitted(rows, systems, fmap)

    # The published scores must fall out of this harness, or the comparison is
    # not on the same footing and nothing below means anything.
    published = json.load(open(os.path.join(RES, 'predictor.json')))['converged_clock']
    for name, key in (('quadrature', 'R1_quadrature'), ('constant', 'R0_null')):
        want = published[key]['mean_of_system_rms']
        got = float(np.mean(list(scores[name].values())))
        if abs(got - want) > 5e-4:
            raise SystemExit(f'{name}: harness gives {got:.4f}, '
                             f'predictor.json says {want:.4f}')

    rng = np.random.default_rng(SEED)
    q, r = scores['quadrature'], scores['ratio_regression']
    diffs = []
    for _ in range(N_RESAMPLE):
        pick = rng.choice(systems, size=len(systems), replace=True)
        diffs.append(float(np.mean([q[s] for s in pick])
                           - np.mean([r[s] for s in pick])))
    diffs = np.array(diffs)
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))

    out = {
        'protocol': 'fitted comparators on the footing of predictor.json',
        'operating_point': 'converged clock',
        'aggregation': 'RMS of the log10 residual within the held-out system, '
                       'then the unweighted mean over systems',
        'n_points': len(rows),
        'rules': {
            name: {
                'coefficients': 0 if name == 'quadrature' else FEATURES[name][0],
                'fitted_leave_one_system_out': name != 'quadrature',
                'per_system': per,
                'mean_of_system_rms': float(np.mean(list(per.values()))),
            } for name, per in scores.items()
        },
        'quadrature_vs_ratio_regression': {
            'systems_where_quadrature_wins':
                sorted(s for s in systems if q[s] < r[s]),
            'difference_mean_of_system_rms':
                float(np.mean(list(q.values())) - np.mean(list(r.values()))),
            'difference_ci95': [lo, hi],
            'resolved': bool(lo > 0 or hi < 0),
            'fraction_of_resamples_favouring_regression':
                float((diffs > 0).mean()),
            'n_resamples': N_RESAMPLE,
            'distinct_multisets': 126,
            'seed': SEED,
        },
    }
    path = os.path.join(RES, 'fitted_baseline.json')
    json.dump(out, open(path, 'w'), indent=1)
    for name, per in scores.items():
        print(f"{name:24s} {np.mean(list(per.values())):.4f}")
    print(f"quad - reg = {out['quadrature_vs_ratio_regression']['difference_mean_of_system_rms']:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  resolved="
          f"{out['quadrature_vs_ratio_regression']['resolved']}")
    print('wrote', os.path.relpath(path, ROOT))


if __name__ == '__main__':
    main()
