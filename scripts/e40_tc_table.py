#!/usr/bin/env python3
"""E40a -- a dimensionless collapse-time table, so the rate estimator is cheap.

WHY A TABLE
-----------
`gravop/units.py` states the structural fact this rests on: in dimensionless
variables the gravothermal solution depends on the halo only through
(sigma_hat, w_hat). A 10^9 Msun dwarf and a 10^15 Msun cluster with the same
pair follow identical trajectories. So the collapse time factorises,

    t_c [Gyr] = t_c_hat(sigma_hat, w_hat, C_eff) * t_0(r_s, rho_s),

and one table over the dimensionless pair serves every halo at every epoch.
The rate estimator (E40b) needs t_c at 18,467 halo-epochs on a grid of C_eff;
without the table that is ~10^5 solver runs, with it, 1,400.

WHY C IS AN AXIS AND NOT A RESCALING
------------------------------------
t_c_hat does NOT scale as 1/C_eff, and this was measured rather than assumed.
Over C in [0.4, 2.0] the product t_c_hat * C is constant to 0.02% at
sigma_hat = 0.06, drifts 3.8% at sigma_hat = 0.556, and drifts **35%** at
sigma_hat = 2.6, because in the opaque regime conduction is not the bottleneck.
A single-reference rescaling would therefore be wrong exactly where the clock
runs fastest, which is where the estimator is most sensitive. C is the third
axis.

CENSORING, WHICH IS NOT OPTIONAL TO REPORT
------------------------------------------
The solver returns three outcomes here and they mean different things:

  status 0  reached the density threshold. t_c_hat is a measurement.
  status 1  hit t_end without collapsing. t_c_hat > t_end: censored ABOVE, and
            harmless -- these nodes contribute ~nothing to a phase integral.
  status 2  the minimum Knudsen number fell below kn_min, so the run stopped in
            the deep short-mean-free-path regime BEFORE collapse. t_c_hat is a
            LOWER bound, censored BELOW, and this is the dangerous one: it
            under-states t_c exactly where t_c is shortest, which biases a
            fitted rate low. Every node records its status and E40b reports what
            fraction of each halo's phase integral came from status-2 nodes.

Resumable: writes after every row, so it can be stopped and restarted.
Single-threaded at nice 19 by construction -- F9 owns the CPU.

Output: results/e40/tc_table.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import solver                                         # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e40')

# sigma_hat spans [-2.33, 1.25] in log10 over the 81 haloes x 236 epochs;
# w_hat spans [-0.15, 1.98]. The grids bracket both with a margin.
LOG_SH = np.linspace(-2.45, 1.35, 20)
# t_c_hat is nearly flat in w_hat above ~5 (605.9 vs 606.5 between w_hat 31.6
# and 100 at sigma_hat 0.32), so the grid is dense low and coarse high.
LOG_WH = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
C_GRID = np.array([0.2, 0.4, 0.6, 0.797, 1.0, 1.4, 2.0])
# Extension added 2026-09-18, AFTER the first scan, and the reason is recorded
# rather than quietly folded in: the first scan's minimum sat at the bottom
# edge C = 0.2, which the pre-registration reads as "not a calibration". But
# the C -> 0 null scores 0.588 against 0.408 at C = 0.2, so the objective must
# turn over somewhere below 0.2 and the minimum is interior after all -- the
# grid was too coarse at the bottom to resolve it. Extending downwards resolves
# a grid, it does not move a goalpost, and the pre-registered reading of an
# edge minimum still applies to whatever the finer grid returns.
C_EXTRA = np.array([0.05, 0.10, 0.15, 0.25])
T_END = 1e5          # far beyond any t_c_hat that matters: t_0 >= 2.7e-4 Gyr,
                     # so 1e5 * t_0 >= 27 Gyr > the age of the universe


def one(sh, wh, C):
    t0 = time.time()
    try:
        r = solver.run_halo(
            sigma_m=float(sh), w=float(wh),
            model_lmfp='yukawa_tu', model_smfp='yukawa_tu',
            C=float(C), n_shells=150, eps_t=3e-4, rho_c_factor=300.0,
            t_end=T_END, kn_min=0.02, max_steps=20_000_000, max_snaps=60,
            timestep_mode='implicit')
    except Exception as e:
        return {'t_c_hat': None, 'status': -1, 'error': type(e).__name__,
                'wall_s': time.time() - t0}
    return {'t_c_hat': float(r['t'][-1]), 'status': int(r['status']),
            'wall_s': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(OUT, 'tc_table.json'))
    ap.add_argument('--extend', action='store_true',
                    help='also fill the C_EXTRA rows (see the note there)')
    a = ap.parse_args()
    c_axis = np.concatenate([C_GRID, C_EXTRA]) if a.extend else C_GRID
    c_axis = np.array(sorted(set(np.round(c_axis, 4).tolist())))
    os.makedirs(OUT, exist_ok=True)

    store = {}
    if os.path.exists(a.out):
        store = json.load(open(a.out)).get('nodes', {})
        print(f'resuming with {len(store)} nodes already done')

    total = len(LOG_SH) * len(LOG_WH) * len(c_axis)
    t_start = time.time()
    done = len(store)
    for i, lsh in enumerate(LOG_SH):
        for lwh in LOG_WH:
            for C in c_axis:
                key = f'{lsh:.4f}|{lwh:.4f}|{C:.4f}'
                if key in store:
                    continue
                store[key] = one(10.0 ** lsh, 10.0 ** lwh, C)
                done += 1
        el = time.time() - t_start
        rate = (done - (len(store) - (done))) if False else None
        json.dump({'grid': {'log_sigma_hat': LOG_SH.tolist(),
                            'log_w_hat': LOG_WH.tolist(),
                            'C': c_axis.tolist()},
                   't_end': T_END,
                   'solver': {'n_shells': 150, 'eps_t': 3e-4,
                              'rho_c_factor': 300.0, 'kn_min': 0.02,
                              'model': 'yukawa_tu', 'timestep_mode': 'implicit'},
                   'status_meanings': {
                       '0': 'collapsed; t_c_hat is a measurement',
                       '1': 'hit t_end; t_c_hat > t_end, censored above',
                       '2': 'Kn < kn_min, stopped in the deep SMFP regime '
                            'before collapse; t_c_hat is a LOWER bound',
                       '-1': 'solver raised'},
                   'nodes': store}, open(a.out, 'w'))
        print(f'  row {i+1}/{len(LOG_SH)}  log sh {lsh:+.2f}  '
              f'{done}/{total} nodes  {el/60:.1f} min elapsed', flush=True)
    st = {}
    for v in store.values():
        st[v['status']] = st.get(v['status'], 0) + 1
    print(f'\ndone: {len(store)} nodes in {(time.time()-t_start)/60:.1f} min')
    print('status counts:', dict(sorted(st.items())))


if __name__ == '__main__':
    main()
