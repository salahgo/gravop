"""Generate cross-domain singularity datasets.

Usage:
  sc_gen.py --system rd --n 600 --seed 0 --out data/sc/rd.h5
  sc_gen.py --all --n 600

Runs are independent, so generation is trivially parallel over processes. Each
dataset stores the fields on the tau grid, the singular observable, both
conditioning variables, the physical parameters and log10 t_sing.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from statecond import data as scd            # noqa: E402
import statecond                                        # noqa: E402
from statecond.pdes import SYSTEMS           # noqa: E402

statecond.load_plugins(statecond.plugin_args_from_argv())

ROOT = os.path.join(os.path.dirname(__file__), '..')


def _one(args):
    name, seed = args
    sys_ = SYSTEMS[name]()
    rng = np.random.default_rng(seed)
    p = sys_.sample(rng)
    tr = sys_.solve(p)
    return tr


def generate(name, n, seed0, workers):
    jobs = [(name, seed0 + i) for i in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        trs = list(ex.map(_one, jobs, chunksize=4))
    wall = time.time() - t0
    n_none = sum(t is None for t in trs)
    n_ns = sum(t is not None and t.status == 1 for t in trs)
    d = scd.build(SYSTEMS[name](), trs)
    return d, {'requested': n, 'failed': n_none, 'nonsingular': n_ns,
               'kept': int(d['u'].shape[0]), 'gen_seconds': round(wall, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'data', 'sc'))
    ap.add_argument('--plugin', action='append', default=[], metavar='MODULE',
                    help='module or .py file registering extra systems (see '
                         'STATECOND_BENCHMARK.md); already imported before '
                         'parsing, this entry documents it')
    args = ap.parse_args()

    names = list(SYSTEMS) if args.all else [args.system]
    os.makedirs(args.outdir, exist_ok=True)
    report = {}
    for k, name in enumerate(names):
        # a distinct seed block per system keeps runs independent and
        # reproducible when systems are generated separately
        d, info = generate(name, args.n, args.seed + 100_000 * k, args.workers)
        path = os.path.join(args.outdir, f'{name}.h5')
        scd.save(path, d)
        info['path'] = os.path.relpath(path, ROOT)
        info['fields'] = int(d['u'].shape[2])
        info['tau_points'] = int(d['u'].shape[1])
        report[name] = info
        print(f"{name:8s} kept {info['kept']}/{info['requested']} "
              f"(failed {info['failed']}, non-singular {info['nonsingular']}) "
              f"in {info['gen_seconds']}s -> {info['path']}", flush=True)

    with open(os.path.join(args.outdir, 'manifest.json'), 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
