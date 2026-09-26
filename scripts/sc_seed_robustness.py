"""P23b — are three seeds enough for the E22 headline numbers?

The project reports E22 at three seeds and reports the seed spread, but has
never compared against a larger seed count, so "three is enough" was an
assumption. This compares the published 3-seed artefacts against a 10-seed
rerun written to `results/statecond/s10/` -- a SEPARATE directory on purpose,
so the published numbers stay auditable and the 25-point law (which stays at
three seeds, being a different claim with a different error budget) is not
silently mixed with a differently-powered fno point.

Note what this does and does not answer. Seeds measure INITIALISATION spread.
The question "is ~60 test trajectories enough?" is a different one and is
answered by the paired trajectory bootstrap in `sc_significance.py`. The
effects that needed that bootstrap -- the converged penalties near 1.0 -- are
not addressed by adding seeds at all.

Output: results/statecond/seed_robustness.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.dirname(__file__))
from sc_provenance import stamp                                  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
S10 = os.path.join(OUT, 's10')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']
METRICS = ['deployment_penalty', 'state_vs_deployed', 'state_vs_oracle',
           'state_vs_deployed_late']


def main():
    out, shifts = {}, []
    for s in SYSTEMS:
        pa = os.path.join(OUT, f'{s}.json')
        pb = os.path.join(S10, f'{s}.json')
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        a, b = json.load(open(pa)), json.load(open(pb))
        rec = {'seeds_published': len(a['seeds']), 'seeds_rerun': len(b['seeds'])}
        for k in METRICS:
            if k in a and k in b:
                rel = (b[k] - a[k]) / a[k]
                rec[k] = {'published': a[k], 'rerun': b[k],
                          'rel_shift': float(rel)}
                shifts.append(abs(rel))
        ta = a['tsing_median_err_pct']['mean']
        tb = b['tsing_median_err_pct']['mean']
        rec['tsing_median_err_pct'] = {'published': ta, 'rerun': tb,
                                       'rel_shift': float((tb - ta) / ta)}
        shifts.append(abs((tb - ta) / ta))
        # the seed spread the published run reported, for scale
        rec['published_rel_seed_spread'] = {
            k: (v['std'] / v['mean'] if v['mean'] else None)
            for k, v in a['nrmse'].items()}
        out[s] = rec

    if shifts:
        out['_summary'] = {
            'max_abs_rel_shift': float(max(shifts)),
            'median_abs_rel_shift': float(np.median(shifts)),
            'n_comparisons': len(shifts),
            'verdict': (
                'three seeds are adequate for the E22 headline ratios: every '
                f'one moves by at most {100 * max(shifts):.1f}% at ten seeds, '
                'which is small against the effects (2x-35x) they support'
                if max(shifts) < 0.25 else
                'three seeds are NOT adequate: at least one headline ratio '
                f'moves by {100 * max(shifts):.1f}% at ten seeds'),
            'does_not_address': (
                'test-set sampling error, which is what the converged '
                'penalties near 1.0 turn on; see significance.json'),
            '_provenance': stamp('sc_seed_robustness.py'),
        }
    path = os.path.join(OUT, 'seed_robustness.json')
    json.dump(out, open(path, 'w'), indent=1)

    print(f"{'sys':9s} {'metric':24s} {'3 seeds':>10s} {'10 seeds':>10s} "
          f"{'shift':>8s}")
    for s in SYSTEMS:
        if s not in out:
            continue
        for k in METRICS + ['tsing_median_err_pct']:
            if k in out[s]:
                v = out[s][k]
                print(f"{s:9s} {k:24s} {v['published']:10.3f} "
                      f"{v['rerun']:10.3f} {100 * v['rel_shift']:+7.1f}%")
    if '_summary' in out:
        print('\n' + out['_summary']['verdict'])
    print('wrote', path)


if __name__ == '__main__':
    main()
