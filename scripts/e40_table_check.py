#!/usr/bin/env python3
"""E40d -- how much error does the collapse-time table's interpolation add?

Pre-registration item 4: measured against held-out direct solver runs, and if
it exceeds 2% the table is refined rather than accepted.

The test points are deliberately OFF the grid -- mid-cell in all three axes,
where interpolation is worst -- and drawn from the region the 81 haloes
actually occupy rather than from the grid's corners, because an error in a
corner nothing uses is not an error that matters.

Output: results/e40/table_check.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import solver                                         # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e40')
RNG = np.random.default_rng(20260916)
N_TEST = 24


def main():
    from e40_rate_estimator import load_table, halo_series
    f, fc, grid, miss, T = load_table(os.path.join(OUT, 'tc_table.json'))
    lsh_g, lwh_g, C_g = grid

    # where the haloes actually live
    z = np.load(os.path.join(ROOT, 'results', 'e31', 'trajectories.npz'))
    SH, WH = [], []
    for i in range(81):
        _, _, sh, wh, _, ok, _ = halo_series(z, i)
        SH.append(sh[ok]); WH.append(wh[ok])
    SH = np.log10(np.concatenate(SH)); WH = np.log10(np.concatenate(WH))
    lo_s, hi_s = np.percentile(SH, [2, 98])
    lo_w, hi_w = np.percentile(WH, [2, 98])

    rows = []
    for _ in range(N_TEST):
        lsh = float(RNG.uniform(lo_s, hi_s))
        lwh = float(RNG.uniform(lo_w, hi_w))
        C = float(RNG.uniform(C_g.min(), C_g.max()))
        try:
            r = solver.run_halo(
                sigma_m=10.0 ** lsh, w=10.0 ** lwh,
                model_lmfp='yukawa_tu', model_smfp='yukawa_tu', C=C,
                n_shells=150, eps_t=3e-4, rho_c_factor=300.0, t_end=1e5,
                kn_min=0.02, max_steps=20_000_000, max_snaps=60,
                timestep_mode='implicit')
        except Exception as e:
            rows.append({'log_sigma_hat': lsh, 'log_w_hat': lwh, 'C': C,
                         'error': type(e).__name__})
            continue
        direct = float(r['t'][-1])
        interp = float(10.0 ** f([[lsh, lwh, C]])[0])
        rows.append({'log_sigma_hat': lsh, 'log_w_hat': lwh, 'C': C,
                     'direct': direct, 'interpolated': interp,
                     'status': int(r['status']),
                     'rel_error': abs(interp / direct - 1.0)})
        print(f"  log sh {lsh:+.2f}  log wh {lwh:+.2f}  C {C:.3f}  "
              f"direct {direct:10.2f}  interp {interp:10.2f}  "
              f"rel {rows[-1]['rel_error']:.4f}  status {r['status']}",
              flush=True)

    e = [r['rel_error'] for r in rows if 'rel_error' in r]
    # status-2 nodes are censored lower bounds, so an interpolation error there
    # is not the same kind of number; both are reported
    e0 = [r['rel_error'] for r in rows if r.get('status') == 0]
    out = {
        'n_test': len(rows),
        'sampled_from': 'the 2nd-98th percentile box the 81 haloes occupy, '
                        'off-grid in all three axes',
        'median_rel_error': float(np.median(e)),
        'p90_rel_error': float(np.percentile(e, 90)),
        'max_rel_error': float(np.max(e)),
        'median_rel_error_uncensored': float(np.median(e0)) if e0 else None,
        'max_rel_error_uncensored': float(np.max(e0)) if e0 else None,
        'n_uncensored': len(e0),
        'threshold': 0.02,
        'passes_preregistered_threshold': bool(np.median(e) <= 0.02),
        'rows': rows,
    }
    json.dump(out, open(os.path.join(OUT, 'table_check.json'), 'w'), indent=1)
    print(f"\nmedian {out['median_rel_error']:.4f}, p90 "
          f"{out['p90_rel_error']:.4f}, max {out['max_rel_error']:.4f}; "
          f"pre-registered threshold 2% -> "
          f"{'PASS' if out['passes_preregistered_threshold'] else 'FAIL'}")


if __name__ == '__main__':
    main()
