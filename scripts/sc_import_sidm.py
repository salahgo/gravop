"""Re-express the gravothermal SIDM dataset in the cross-domain schema.

The point of the cross-domain study is a single table in which SIDM is one row
among several, measured by the same protocol as the others. That only works if
the SIDM data goes through the same code path, so this converts `data/d1` into
the `statecond.data` schema rather than comparing numbers computed by two
different pipelines.

Mapping:
    fields      (log10 rho, log10 v)          -> u, two fields
    parameters  (log10 sigma0, log10 w, model)-> par
    t_sing      t_c, the collapse time         -> log_tsing
    observable  A = rho_c, the central density -> a
                so phi = log10(rho_c/rho_c(0)), which is exactly the phase
                variable E20 defined for the haloes

Runs whose termination time is solution-dependent are kept: status 0 (collapse)
and status 2 (cut at the fluid-validity boundary), 603 of 1300. The 697 status-1
runs are dropped because they stop at the fixed integration horizon, so their
log t_norm is a constant -- audit finding A6, and the one population for which a
scalar time head has nothing to learn. This matches the other four systems,
whose parameter boxes are chosen so that every retained run reaches its
threshold.

Usage:  sc_import_sidm.py [--src data/d1/train.h5] [--out data/sc/sidm.h5]
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from statecond import data as scd            # noqa: E402
from gravop import dataset as gds            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')


def require_source(path):
    """A clean clone has no `data/d1`, and the raw h5py error does not say so.

    R5 (2026-09-08) ran the documented from-scratch path and this step failed
    with a FileNotFoundError raised four frames deep inside h5py. The path was
    in the message, but nothing said that the file is a *generated* artefact,
    that it is gitignored by design, or which command makes it.
    """
    if os.path.exists(path):
        return
    raise SystemExit(
        f'{path} not found.\n\n'
        'data/d1 is generated, not committed -- it is ~8 min of solver time '
        'and is gitignored by design. Make it first:\n\n'
        '    python scripts/gen_dataset.py train 1300 --seed 20260827\n\n'
        'then re-run this script. The four bespoke PDE systems need none of '
        'this; only `sidm` is imported from the gravothermal training set.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(ROOT, 'data', 'd1', 'train.h5'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'sc', 'sidm.h5'))
    args = ap.parse_args()
    require_source(args.src)

    with h5py.File(args.src, 'r') as f:
        keys = ('lrho', 'lv', 'lrho0', 'lv0', 'log_sigma0', 'log_w',
                'model_id', 'log_tc', 'status')
        d = {k: f[k][:] for k in keys}

    keep = d['status'] != 1              # solution-dependent termination only
    for k in d:
        d[k] = d[k][keep]
    n_runs = d['lrho'].shape[0]

    u = np.stack([d['lrho'], d['lv']], axis=2).astype(np.float32)
    # A = central density; the innermost grid point, as in E20
    a = (10.0 ** d['lrho'][:, :, 0]).astype(np.float64)
    phi, flag = scd.phase_variables(a)
    par = np.stack([d['log_sigma0'], d['log_w'],
                    d['model_id'].astype(np.float32)], 1).astype(np.float32)

    out = {
        'u': u,
        'u0': u[:, 0].copy(),
        'a': a.astype(np.float32),
        'phi': phi,
        'flag': flag,
        'par': par,
        'log_tsing': d['log_tc'].astype(np.float32),
        'status': np.zeros(n_runs, np.int8),   # all singular by construction
        'wall': np.zeros(n_runs, np.float32),
        'x': np.log10(gds.r_grid()).astype(np.float32),
        'param_names': ['log_sigma0', 'log_w', 'model_id'],
        'system': 'sidm',
    }
    assert u.shape[1] == scd.N_TAU, (u.shape, scd.N_TAU)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scd.save(args.out, out)
    print(f"sidm: kept {n_runs} runs with solution-dependent t_norm "
          f"(of {keep.size}) -> {os.path.relpath(args.out, ROOT)}")
    print(f"  fields {u.shape[2]}, grid {u.shape[3]}, tau points {u.shape[1]}, "
          f"params {par.shape[1]}")
    print(f"  log10 t_c spread {d['log_tc'].max() - d['log_tc'].min():.2f} decades")


if __name__ == '__main__':
    main()
