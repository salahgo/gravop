#!/usr/bin/env python3
"""Energy conservation of the NSphere runs up to the collapse marker.

NSphere writes total_energy_vs_time_*.dat (float32 records: time in Myr,
kinetic, potential, total energy, relative drift of the total energy). For
every run with a data directory this script records the absolute relative
drift at the run's t_x10 (run.json, markers_t0; linear interpolation in time),
its maximum over every record up to t_x10, and its maximum over the whole run. The earlier B and C runs
(results/independent_halo_20261107) take t_x10 from their measures and t0 from
the halo definition. Runs whose raw data are no longer on disk keep their
stored entry or are absent. Writes closure_design/energy_drift.json.
"""
import glob
import json
import os

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]
import independent_halo_analysis as IH  # noqa: E402
OUT = os.path.join(EXT, 'closure_design', 'energy_drift.json')


def drift(a, t_myr):
    pre = np.abs(a[a[:, 0] <= t_myr, 4])
    return dict(drift_at_tx10=float(abs(np.interp(t_myr, a[:, 0], a[:, 4]))),
                max_drift_to_tx10=float(max(pre.max() if pre.size else 0.0,
                                            abs(np.interp(t_myr, a[:, 0], a[:, 4])))),
                max_drift=float(np.abs(a[:, 4]).max()))


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for grp in ('closure_design', 'multifidelity'):
        for d in sorted(glob.glob(os.path.join(EXT, grp, 'nsphere', '*'))):
            f = glob.glob(os.path.join(d, 'data', 'total_energy_vs_time_*.dat'))
            rj = os.path.join(d, 'run.json')
            mz = os.path.join(d, 'measures.npz')
            if not f or not os.path.exists(rj) or not os.path.exists(mz):
                continue
            tx = json.load(open(rj)).get('markers_t0', {}).get('t_x10')
            if tx is None:
                continue
            a = np.fromfile(f[0], dtype='<f4').reshape(-1, 5)
            t_myr = tx * float(np.load(mz)['t0_myr'])
            out[os.path.basename(d)] = drift(a, t_myr)
    pub = os.path.join(ROOT, 'results', 'independent_halo_20261107')
    for d in sorted(glob.glob(os.path.join(pub, 'nsphere', '*'))):
        run = os.path.basename(d)
        f = glob.glob(os.path.join(d, 'data', 'total_energy_vs_time_*.dat'))
        mz = os.path.join(pub, f'nsphere_{run}.npz')
        if not f or not os.path.exists(mz):
            continue
        m = np.load(mz)
        tx = IH.collapse_markers(m['t'], m['mean_rho'][:, 1])['t_x10']
        a = np.fromfile(f[0], dtype='<f4').reshape(-1, 5)
        t_myr = tx * IH.halo_units(run[0])['t0_myr']
        out[run] = drift(a, t_myr)
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, v in out.items():
        print(f'{k:14s} {100 * v["drift_at_tx10"]:.3f}%  to marker {100 * v.get("max_drift_to_tx10", float("nan")):.3f}%  max {100 * v["max_drift"]:.1f}%')


if __name__ == '__main__':
    main()
