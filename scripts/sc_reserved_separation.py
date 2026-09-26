#!/usr/bin/env python3
"""How far the reserved draws actually sit from the draws that were trained on.

WHY THIS EXISTS
---------------
The prospective test says the reserved draws are "disjoint from every split
used here" and "never used to train, tune, select or score anything". Both are
true of the draws as objects. Neither says how far they are from the training
set in parameter space, and a reserved draw that lands on top of a training
draw is a weaker test than one that does not, because the operator is
unusually accurate there.

Measured here, once, so the manuscript can report the distribution rather than
assert disjointness. Separations are L1 in units of the training set's own
per-coordinate standard deviation, so they are comparable across systems whose
parameters have different ranges and dimensions.

WHAT IT FINDS
-------------
Four systems sit at median separations of 0.3 to 0.8 training standard
deviations, which is a genuine gap. `sidm` is the exception and the reason is
structural rather than a defect: its constant-cross-section branch sets w to a
fixed value, so those draws live on a line in an otherwise two-dimensional box.
207 training draws and 17 reserved draws on one line leave small gaps by
density alone, and eight reserved draws land within a hundredth of a standard
deviation of a training draw. Every one of those eight is on the constant
branch; the velocity-dependent branch keeps a median separation of 0.21.

The quantity the prospective test scores is a ratio of the operator's own
errors at the same coordinate, so an unusually accurate point moves numerator
and denominator together. The effect is second order, which is the reading the
manuscript takes -- and it is reported rather than left for a reader to find.
"""
import json
import os

import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
SYSTEMS = ('rd', 'ks', 'film', 'burgers', 'sidm')
NEAR = 0.01          # "on top of a training draw", in training-sd units


def main():
    draws = json.load(open(os.path.join(ROOT, 'results', 'e43',
                                        'reserved_draws.json')))
    out = {
        'metric': 'L1 distance to the nearest training draw, per coordinate '
                  'in units of the training set standard deviation',
        'near_threshold': NEAR,
        'systems': {},
    }
    for name in SYSTEMS:
        with h5py.File(os.path.join(ROOT, 'data', 'sc', f'{name}.h5'), 'r') as h:
            train = np.asarray(h['par'])
        reserved = np.array([d['vector'] for d in draws
                             if d['system'] == name
                             and d['role'] == 'reserved_confirmation'])
        sd = train.std(axis=0)
        sd[sd == 0] = 1.0
        dist = np.abs((reserved[:, None, :] - train[None, :, :]) / sd).sum(axis=2)
        nearest = dist.min(axis=1)
        rec = {
            'n_reserved': int(len(reserved)),
            'n_train': int(len(train)),
            'nearest_training_separation': {
                'min': float(nearest.min()),
                'p5': float(np.percentile(nearest, 5)),
                'median': float(np.median(nearest)),
                'max': float(nearest.max()),
            },
            'n_within_threshold': int((nearest < NEAR).sum()),
        }
        if name == 'sidm':
            # The third coordinate is the cross-section branch flag. On the
            # constant branch the velocity scale is held fixed, so the draws
            # occupy a line rather than the box.
            const = reserved[:, 2] == 0
            rec['constant_branch'] = {
                'n_reserved': int(const.sum()),
                'n_train': int((train[:, 2] == 0).sum()),
                'median_separation': float(np.median(nearest[const])),
            }
            rec['velocity_dependent_branch'] = {
                'n_reserved': int((~const).sum()),
                'median_separation': float(np.median(nearest[~const])),
            }
            rec['all_near_draws_are_on_the_constant_branch'] = \
                bool((~const[nearest < NEAR]).sum() == 0)
        out['systems'][name] = rec
        print(f"{name:8s} n={rec['n_reserved']:3d}  median={rec['nearest_training_separation']['median']:.3f}"
              f"  min={rec['nearest_training_separation']['min']:.3f}"
              f"  within {NEAR}: {rec['n_within_threshold']}")

    path = os.path.join(ROOT, 'results', 'e43', 'reserved_separation.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', os.path.relpath(path, ROOT))


if __name__ == '__main__':
    main()
