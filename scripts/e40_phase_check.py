#!/usr/bin/env python3
"""E40c -- is the fitted phase independent of C_eff?

The rate estimator compares a MODELLED phase, which depends on C_eff, against
the phase E32 fitted, which was obtained at C_eff = 0.797. If that fitted phase
moved with C_eff the comparison would be circular: both sides would carry the
parameter being estimated and the scan could find a minimum that means nothing.

Physically it should not move -- C_eff rescales the clock, while the phase is a
position along the track in (vmax, rmax), which is a shape. This measures it
rather than asserting it, on haloes spanning the observed phase range, and
records the residual dependence as an uncertainty the rate result must carry.

Output: results/e40/phase_c_independence.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import e30_concerto as e30                                        # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e40')
C_TEST = (0.4, 0.797, 2.0)


def main():
    z = np.load(os.path.join(ROOT, 'results', 'e31', 'trajectories.npz'))
    ctl = np.load(os.path.join(ROOT, 'results', 'e32', 'controls.npz'))
    ph = ctl['phase']
    mov = np.where(ph > 0.02)[0]
    order = mov[np.argsort(ph[mov])]
    pick = order[np.linspace(0, len(order) - 1, 10).astype(int)]

    rows = []
    for i in pick:
        vals = {}
        for C in C_TEST:
            r = e30.one_halo(float(z['cdm_vmax_z0'][i]),
                             float(z['cdm_rmax_z0'][i]),
                             float(z['sidm_vmax_z0'][i]),
                             float(z['sidm_rmax_z0'][i]),
                             c_eff=C, n_shells=150)
            vals[f'{C}'] = None if r is None else float(r['best_phase'])
        v = [x for x in vals.values() if x is not None]
        rows.append({'halo': int(i), 'phase_by_C': vals,
                     'spread': float(max(v) - min(v)) if len(v) > 1 else None,
                     'e32_phase': float(ph[i]),
                     'reproduces_e32': bool(vals.get('0.797') is not None
                                            and abs(vals['0.797']
                                                    - float(ph[i])) < 1e-6)})
        print(f"halo {i:3d}  " + '  '.join(
            f'C={k}: {x:.4f}' for k, x in vals.items() if x is not None)
            + f"   spread {rows[-1]['spread']:.4f}   E32 {ph[i]:.4f}"
            + ('  [reproduces]' if rows[-1]['reproduces_e32'] else '  [DIFFERS]'))

    sp = [r['spread'] for r in rows if r['spread'] is not None]
    out = {
        'question': 'does the fitted phase move with C_eff? if it did, the '
                    'rate scan would be circular',
        'C_tested': list(C_TEST),
        'n_haloes': len(rows),
        'max_spread': float(max(sp)),
        'median_spread': float(np.median(sp)),
        'spread_below_phase_0p6': float(max(
            r['spread'] for r in rows if r['e32_phase'] < 0.6)),
        'all_reproduce_e32_at_0p797': bool(all(r['reproduces_e32'] for r in rows)),
        'reading': 'the phase is a position along the track, which is a shape; '
                   'C_eff rescales the clock. The residual dependence is the '
                   'uncertainty the rate result carries, and it is largest '
                   'near phase 1 where the track is steepest.',
        'rows': rows,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'phase_c_independence.json'), 'w'),
              indent=1)
    print(f"\nmax spread {out['max_spread']:.4f}, median "
          f"{out['median_spread']:.4f}, below phase 0.6 "
          f"{out['spread_below_phase_0p6']:.4f}; "
          f"all reproduce E32 at C=0.797: {out['all_reproduce_e32_at_0p797']}")


if __name__ == '__main__':
    main()
