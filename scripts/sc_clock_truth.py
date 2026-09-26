"""P21 (concluded) — what the converged clock is worth against TRUTH.

THE FINDING THIS EXISTS TO QUANTIFY
-----------------------------------
`sc_refine_reference.py` refined the reference `t_sing` from a 2x to a 4x grid
with Richardson extrapolation, and the production error did not fall -- it rose
slightly (`ks` 7.81e-04 vs 2x, 1.04e-03 vs the Richardson limit), because the 2x
comparison was partially cancelling the reference's own error against the
production grid's. The observed order is a clean 2.00 in all four systems, so
the schemes are behaving; **the limit is the PRODUCTION grid, not the
reference**, and no amount of reference refinement moves it.

That matters because of what the head is measured against. The dataset's
`log_tsing` is the production-grid value. The converged head (P20) reproduces
THAT to 0.037%-0.82%. Its error against the TRUE singular time is the
combination of two independent terms:

    (head vs label)  +  (label vs truth, i.e. the production grid error)

and in three of four reference systems the second term is the larger one. So
"the converged head reaches 0.037%" is a statement about reproducing the
benchmark's labels, not about recovering the singular time.

WHAT THIS CHANGES, AND WHAT IT DOES NOT
---------------------------------------
It does NOT invalidate the deployment-penalty measurements. Arm B is queried at
`tau = t / t_sing_hat` where both the head and the evaluation targets are
defined on the same production labels, so a common offset cancels and the
penalty is a valid measurement of what a clock error of that size costs.

It DOES change the interpretation. The converged penalty (0.97-2.10x) is the
penalty at the head's error against the labels. A real deployment faces the
grid error too, so the effective clock error is larger and the true penalty is
higher. This script computes how much, using the clock-error floor F(d) --
which needs no network and is exactly the tool for the question.

Output: results/statecond/clock_truth.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from statecond import data as scd                                 # noqa: E402
from sc_experiment import Norm, LATE                              # noqa: E402
from sc_displacement import resample_tau                                 # noqa: E402
from sc_provenance import stamp                                   # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ('rd', 'ks', 'film', 'burgers')     # reference systems only


def floor_at(name, deltas):
    """F(d) for arbitrary d, in the units `field_err` reports in.

    Same construction as `sc_displacement.floor_for_system`: symmetrised over sign,
    on the test split, in the training-split normalisation.
    """
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    un = (d['u'] - n.f_mu) / n.f_sd
    taus = scd.tau_grid()
    late = taus > LATE
    out = {}
    for delta in deltas:
        acc, acc_late = [], []
        for sgn in (+1.0, -1.0):
            tq = np.clip(taus / (1.0 + sgn * delta), 0.0, 1.0)
            up = resample_tau(un, taus, tq, 'linear')
            e = np.sqrt(((up[te] - un[te]) ** 2).mean(axis=(2, 3)))
            acc.append(float(e.mean()))
            acc_late.append(float(e[:, late].mean()))
        out[delta] = {'F': float(np.mean(acc)),
                      'F_late': float(np.mean(acc_late))}
    return out


def main():
    ref = json.load(open(os.path.join(OUT, 'refined_reference.json')))
    th = json.load(open(os.path.join(OUT, 'timehead.json')))
    out = {}
    for name in SYSTEMS:
        grid = ref[name]['rel_vs_richardson_median']          # label vs truth
        head = th[name]['converged_head_err_pct'] / 100.0     # head vs label
        # independent error sources, so they combine in quadrature
        total = float(np.hypot(head, grid))
        E_A = th[name]['armA']
        F = floor_at(name, [head, total])
        rec = {
            'head_vs_label': head,
            'label_vs_truth_grid': grid,
            'effective_vs_truth': total,
            'grid_dominates': bool(grid > head),
            'inflation_of_clock_error': total / head,
            'armA_oracle_error': E_A,
            'floor_at_head_error': F[head]['F'],
            'floor_at_effective_error': F[total]['F'],
            # sqrt(E_A^2 + F^2) is the relation P20 verified at the converged
            # operating point (median ratio 0.97); reuse it, do not refit
            'penalty_predicted_at_head_error':
                float(np.hypot(E_A, F[head]['F']) / E_A),
            'penalty_predicted_at_effective_error':
                float(np.hypot(E_A, F[total]['F']) / E_A),
            'penalty_measured_converged': th[name]['penalty_converged'],
        }
        rec['penalty_understated_by'] = (
            rec['penalty_predicted_at_effective_error']
            / rec['penalty_predicted_at_head_error'])
        out[name] = rec
        print(f"{name:9s} head {100*head:7.4f}%  grid {100*grid:7.4f}%  "
              f"-> effective {100*total:7.4f}% ({rec['inflation_of_clock_error']:5.2f}x)"
              f" | penalty {rec['penalty_predicted_at_head_error']:5.3f} -> "
              f"{rec['penalty_predicted_at_effective_error']:5.3f} "
              f"({rec['penalty_understated_by']:5.3f}x)"
              f" {'[GRID DOMINATES]' if rec['grid_dominates'] else ''}")

    inf = [out[k]['inflation_of_clock_error'] for k in out]
    und = [out[k]['penalty_understated_by'] for k in out]
    dom = [k for k in out if out[k]['grid_dominates']]
    out['_summary'] = {
        'clock_error_inflation_range': [min(inf), max(inf)],
        'penalty_understatement_range': [min(und), max(und)],
        'grid_dominates_in': sorted(dom),
        'verdict': (
            'the converged head is measured against production-grid labels, '
            f'and in {len(dom)} of {len(out)} reference systems the grid error '
            'is the LARGER term. Against the true singular time the clock '
            f'error is {min(inf):.2f}-{max(inf):.2f}x what P20 reports, and '
            'the floor predicts the deployment penalty is understated by '
            f'{min(und):.3f}-{max(und):.3f}x. The penalty MEASUREMENTS remain '
            'valid -- head and targets share the same labels, so a common '
            'offset cancels -- but the converged operating point is optimistic '
            'as a description of deployment.'),
        '_provenance': stamp('sc_clock_truth.py'),
    }
    path = os.path.join(OUT, 'clock_truth.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('\n' + out['_summary']['verdict'])
    print('wrote', path)


if __name__ == '__main__':
    main()
