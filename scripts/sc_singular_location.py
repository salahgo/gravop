"""Is the singular LOCATION already solution-dependent in the 1-D benchmark?

WHY THIS EXISTS
---------------
The standing objection to this benchmark is that all four PDE testbeds and the
application are one-dimensional.
The sharpest version of it is not about the dimension count but about a
STRUCTURAL degree of freedom: in a radially symmetric problem the singularity
forms at a known point, so an operator conditioned on a global scalar never has
to say *where*. In 2-D the aggregate forms wherever the initial data sends it.

That is the one thing a 2-D testbed would add that nothing here does, so
whether it is already covered is a measurement, not an opinion. For each
system, this reports where the singular feature sits at the last stored instant,
across the whole dataset, using each system's own definition of the feature.

Usage:  sc_singular_location.py
Output: results/statecond/singular_location.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import statecond.data as scd                                   # noqa: E402
from sc_provenance import stamp                                # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
DATA = os.path.join(ROOT, 'data', 'sc')

# how each system's singular feature is located in its stored field, matching
# the `observable` each System defines
FEATURE = {
    'rd': ('argmax', 'peak of u, which is what observable() maximises'),
    'ks': ('argmax', 'radial: concentration at r -> 0 by symmetry'),
    'sidm': ('argmax', 'radial: core collapse at r -> 0 by symmetry'),
    'film': ('argmin', 'rupture is the MINIMUM of h; observable is 1/min h'),
    'burgers': ('argmax_grad', 'shock is the steepest gradient'),
}


def locate(name, last):
    kind = FEATURE[name][0]
    if kind == 'argmin':
        return np.argmin(last, axis=1)
    if kind == 'argmax_grad':
        return np.argmax(np.abs(np.diff(last, axis=1)), axis=1)
    return np.argmax(last, axis=1)


def main():
    out = {}
    for name in ('rd', 'ks', 'film', 'burgers', 'sidm'):
        d = scd.load(os.path.join(DATA, f'{name}.h5'))
        u = np.asarray(d['u'], dtype=np.float64)
        last = u[:, -1, 0, :]
        n_x = last.shape[1]
        loc = locate(name, last)
        frac = loc / n_x
        out[name] = {
            'n_runs': int(last.shape[0]),
            'n_x': int(n_x),
            'feature': FEATURE[name][1],
            'distinct_sites': int(len(np.unique(loc))),
            'median_fraction': float(np.median(frac)),
            'iqr_fraction': [float(np.percentile(frac, 25)),
                             float(np.percentile(frac, 75))],
            'std_fraction': float(frac.std()),
            'location_is_solution_dependent': bool(len(np.unique(loc)) > 1),
        }
    fixed = sorted(k for k, v in out.items()
                   if not v['location_is_solution_dependent'])
    varying = sorted(k for k, v in out.items()
                     if v['location_is_solution_dependent'])
    out['_summary'] = {
        'fixed_by_symmetry': fixed,
        'solution_dependent': varying,
        'most_variable': max(varying, key=lambda k: out[k]['distinct_sites']),
        'reading': (
            'The singular location is fixed by symmetry in '
            f'{len(fixed)} of 5 systems ({fixed}) and solution-dependent in '
            f'{len(varying)} ({varying}). So the structural objection a 2-D '
            'testbed answers is PARTLY covered already, and what 2-D would '
            'add is competing and merging peaks rather than a single feature '
            'that moves.'),
        '_provenance': stamp('sc_singular_location.py'),
    }
    path = os.path.join(OUT, 'singular_location.json')
    json.dump(out, open(path, 'w'), indent=1)
    for k in ('rd', 'ks', 'film', 'burgers', 'sidm'):
        v = out[k]
        print(f"{k:8s} {v['distinct_sites']:3d} distinct sites / {v['n_x']}  "
              f"median {v['median_fraction']:.3f}  "
              f"IQR [{v['iqr_fraction'][0]:.3f}, {v['iqr_fraction'][1]:.3f}]  "
              f"{'SOLUTION-DEPENDENT' if v['location_is_solution_dependent'] else 'fixed by symmetry'}")
    print('\n' + out['_summary']['reading'])
    print('wrote', path)


if __name__ == '__main__':
    main()
