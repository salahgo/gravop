"""Exact scalar benchmark for the outer parameter-CDF transition found in F6."""
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from benchmark_conditional_outer import adaptive


def main():
    center, width = .66, .016
    lo, hi = -1., 1.5
    function = lambda y: ndtr((y-center)/width)

    def primitive(y):
        z = (y-center)/width
        return width*(z*ndtr(z)+np.exp(-z*z/2)/np.sqrt(2*np.pi))

    reference = primitive(hi)-primitive(lo)
    base = [lo, .25, .6492923469631748, hi]
    features = [center-6*width, center, center+6*width]
    rows = []
    for tolerance in (.02, .005, .00125, .0001):
        for method, cuts in [('original_cuts', base), ('known_feature_cuts', sorted(base+features))]:
            row = adaptive(function, cuts, tolerance)
            row.update(method=method, tolerance=tolerance, reference=reference,
                       actual_relative_error=abs(row['value']-reference)/reference)
            rows.append(row)
            print(method, tolerance, row['nodes'], row['embedded_relative_error'], row['actual_relative_error'])
    out = dict(scope='Constructed Gaussian parameter-CDF transition, exact antiderivative; no SIDM solves; known-feature discovery cost excluded',
               center=center, width=width, cases=rows)
    path = Path(__file__).resolve().parents[1]/'results/forecast_conditional/outer_parameter_cdf_benchmark.json'
    path.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
