#!/usr/bin/env python3
"""E43b -- the SIDM half of the reserved reference set. Nothing is scored here.

Protocol: docs/PROSPECTIVE_PREREGISTRATION.md.

SIDM is separated from the other four because its reserved records carry the
initial condition as a NAME plus a shape parameter (`ic` in {nfw, cored, gnfw},
`rc`), while the training set stores an integer `ic_type`. The pre-registration
requires that mapping to be established from the generator rather than guessed.
It is, at `gen_dataset.py:109`:

    {'nfw': 0, 'cored': 1, 'hernquist': 2, 'gnfw': 3}

and the training set contains exactly {0, 1, 3} -- no hernquist, which is the
out-of-family set. So the reserved draws use the same three families the
operators were trained on, and `rc` means what it means there: a core radius for
`cored` and an inner slope for `gnfw`.

This reuses `gen_dataset.make_ic` and the frozen `SOLVER_KW` rather than
reimplementing either, so the reserved references are produced by the same code
path as the training data. The only difference is that parameters are READ from
the reservation instead of drawn from an rng.

Output: data/sc/reserved/sidm_full.h5, then the statecond re-expression.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from gravop import solver, dataset                                # noqa: E402

spec = importlib.util.spec_from_file_location(
    'gen_dataset', os.path.join(os.path.dirname(__file__), 'gen_dataset.py'))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

OUT = os.path.join(ROOT, 'data', 'sc', 'reserved')
IC_CODE = {'nfw': 0, 'cored': 1, 'hernquist': 2, 'gnfw': 3}


def one(rec):
    p = {k: (float(v) if k in ('sigma_m', 'w') else v)
         for k, v in rec['params'].items()}
    ic, rc = rec['ic'], rec.get('rc')
    t0 = time.time()
    prof = gen.make_ic(ic, rc, gen.SOLVER_KW['n_shells'])
    kw = dict(gen.SOLVER_KW)
    if prof is not None:
        kw['r'], kw['m'], kw['rho'], kw['p'] = prof
    res = solver.run_halo(**p, **kw)
    if res['status'] in (3, 4) or res['t_final'] < 1.0:
        raise RuntimeError(f"unusable: status {res['status']} t {res['t_final']}")
    s = dataset.sample_from_run(res)
    s['ic_type'] = IC_CODE[ic]
    s['rc'] = np.float32(rc if rc else 0.0)
    s['seed'] = rec['seed']
    s['wall'] = np.float32(time.time() - t0)
    return s


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    from e43_prospective_generate import load_reserved, DRAWS
    rows = [r for r in load_reserved(DRAWS) if r['system'] == 'sidm']
    assert len(rows) == 64, len(rows)
    assert set(r['ic'] for r in rows) <= {'nfw', 'cored', 'gnfw'}, (
        'a reserved SIDM case uses an initial-condition family the operators '
        'were not trained on; that is an out-of-family test, not this one')

    os.makedirs(OUT, exist_ok=True)
    samples, failed = [], []
    t0 = time.time()
    for r in rows:
        try:
            samples.append(one(r))
        except Exception as e:
            failed.append({'id': r['id'], 'error': repr(e)[:200]})
            print(f"  {r['id']}: {type(e).__name__}", flush=True)
    wall = time.time() - t0
    print(f'sidm     {len(samples):3d}/{len(rows)} usable  {wall:6.1f}s')

    if samples:
        keys = [k for k in samples[0]]
        d = {k: np.stack([s[k] for s in samples]) for k in keys}
        import h5py
        with h5py.File(os.path.join(OUT, 'sidm_full.h5'), 'w') as f:
            for k, v in d.items():
                f.create_dataset(k, data=v)
    json.dump({'reserved': len(rows), 'solved': len(samples),
               'failed': failed, 'wall_s': wall,
               'ic_mapping_source': 'gen_dataset.py:109',
               'ids': [r['id'] for r in rows]},
              open(os.path.join(OUT, 'sidm_generation.json'), 'w'), indent=1)
    print('wrote data/sc/reserved/sidm_full.h5  (no scoring performed)')


if __name__ == '__main__':
    main()
