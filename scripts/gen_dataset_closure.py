#!/usr/bin/env python3
"""Dataset D1 regenerated under another conductivity closure.

Runs scripts/gen_dataset.py unchanged (same parameter draws, seeds, initial
profiles, solver settings and HDF5 format) with its solver call routed through
gravop.closure.run_halo_g at the given (alpha, C). With the production base
seed and 1300 runs, every trajectory is paired with the one of the same index
in data/d1/train.h5.

Usage: gen_dataset_closure.py <name> <n_runs> --alpha 0.5 --C 0.9 [--workers W]
Output: data/d1/<name>.h5 (attributes record alpha and C)
"""
import argparse
import os
import sys
import types

import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)
from gravop import closure, solver  # noqa: E402
import gen_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('n_runs', type=int)
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--C', type=float, required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260827)
    a = ap.parse_args()
    shim = types.SimpleNamespace(
        run_halo=lambda **kw: closure.run_halo_g(alpha=a.alpha, C=a.C, **kw),
        profile_from_rho=solver.profile_from_rho, cored_nfw_rho=solver.cored_nfw_rho)
    gen_dataset.solver = shim          # inherited by the forked workers
    sys.argv = ['gen_dataset.py', a.name, str(a.n_runs), '--seed', str(a.seed),
                '--workers', str(a.workers)]
    gen_dataset.main()
    path = os.path.join(HERE, '..', 'data', 'd1', a.name + '.h5')
    with h5py.File(path, 'a') as h:
        h.attrs['closure_alpha'] = a.alpha
        h.attrs['closure_C'] = a.C


if __name__ == '__main__':
    main()
