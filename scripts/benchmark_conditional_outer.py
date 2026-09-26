"""Analytic feature benchmark for a future outer-integrator revision.

No solver evaluations or model changes. Breakpoints here are known from the
constructed functions; finding them in SIDM has an additional, unmeasured cost.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gravop.dyadic import dyadic_nodes


def adaptive(function, cuts, tolerance, cap=257):
    roots = list(zip(cuts[:-1], cuts[1:]))
    cache = {}

    def panel(root, index, depth):
        nodes = dyadic_nodes(roots[root], index, depth)
        if len(cache)+sum(float(y) not in cache for y in nodes) > cap:
            raise RuntimeError('node cap')
        for y in nodes:
            if float(y) not in cache:
                cache[float(y)] = float(function(y))
        values = np.array([cache[float(y)] for y in nodes])
        span = nodes[-1]-nodes[0]
        fine = span*np.dot([1, 4, 2, 4, 1], values)/12
        coarse = span*np.dot([1, 4, 1], values[::2])/6
        return dict(root=root, index=index, depth=depth, value=fine, error=abs(fine-coarse))

    panels = [panel(i, 0, 0) for i in range(len(roots))]
    reason = 'tolerance'
    while True:
        value = sum(p['value'] for p in panels)
        error = sum(p['error'] for p in panels)
        if error <= tolerance*value:
            break
        k = int(np.argmax([p['error'] for p in panels]))
        p = panels[k]
        try:
            children = [panel(p['root'], 2*p['index']+j, p['depth']+1) for j in (0, 1)]
        except RuntimeError:
            reason = 'node cap'
            break
        panels[k:k+1] = children
    return dict(value=value, embedded_relative_error=error/value, nodes=len(cache),
                panels=len(panels), reason=reason)


def main():
    base = [-1., .25, .6492923469631748, 1.5]

    def truncated_ridge(y):
        center = -.33+1.33*np.exp(-2*(y+.56))
        return ndtr((1-center)/.004)-ndtr((-1.5-center)/.004)

    def cdf_birth(y):
        peak_minus_threshold = .2*np.tanh(2*(y-.1))
        fraction = 1. if peak_minus_threshold <= 0 else 2*ndtr(-np.sqrt(2*peak_minus_threshold))
        return np.exp(peak_minus_threshold)*fraction

    shoulder = 6*.004/(2*1.33)
    cases = [('prior_boundary', truncated_ridge, [-.56-shoulder, -.56, -.56+shoulder]),
             ('likelihood_cdf_birth', cdf_birth, [.1])]
    rows = []
    for name, function, features in cases:
        reference, reference_error = quad(function, -1., 1.5, points=features,
                                           epsabs=1e-11, epsrel=1e-11, limit=300)
        assert reference_error < 1e-9*reference
        for tolerance in (.02, .005, .00125, .0001):
            for method, cuts in [('original_cuts', base), ('known_feature_cuts', sorted(base+features))]:
                result = adaptive(function, cuts, tolerance)
                result.update(case=name, method=method, tolerance=tolerance,
                    reference=reference, reference_absolute_error=reference_error,
                    actual_relative_error=abs(result['value']-reference)/reference)
                rows.append(result)
    out = dict(scope='Analytic scalar outer integrals, not SIDM validation; known-feature discovery cost excluded',
        sources=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html',
                 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.nquad.html'],
        cases=rows)
    path = ROOT/'results/forecast_conditional/outer_feature_benchmark.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2)+'\n')
    for row in rows:
        print(row['case'], row['tolerance'], row['method'], 'nodes', row['nodes'],
              'estimated', f"{row['embedded_relative_error']:.3g}",
              'actual', f"{row['actual_relative_error']:.3g}", row['reason'])


if __name__ == '__main__':
    main()
