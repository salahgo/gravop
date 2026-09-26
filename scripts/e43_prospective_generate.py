#!/usr/bin/env python3
"""E43a -- generate the reserved confirmation references. Nothing is scored here.

Protocol: docs/PROSPECTIVE_PREREGISTRATION.md, committed before this ran.

Deliberately split from the scoring step. The generator writes reference
trajectories and stops; it computes no error, loads no network and reports no
predictor number. That separation is the point: it makes it impossible to see a
score while still deciding how to build the data.

The 320 reserved draws are a deterministic function of their stored parameters
-- `System.initial(p)` and `System.rhs` consume `params` and nothing else -- so
no seed is needed and regeneration is exact. Verified against a training case
before this script was written: `rd[0]` reproduces its log10 t_sing to 9.7e-9.

The four benchmark systems are handled here. SIDM is not: its reserved records
carry `ic` as a name and an `rc`, against an integer `ic_type` in the training
set, and the pre-registration requires that mapping to be established from the
generator rather than guessed. It is handled separately or excluded with that
reason recorded.

Output: data/sc/reserved/<system>.h5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
import statecond.pdes as pdes                                     # noqa: E402
import statecond.data as scd                                      # noqa: E402

# The frozen reservation lives IN the repository, not at a path on the machine
# that produced it: it is an input to a published result, so a reader who cannot
# read it cannot reproduce E43. Its digest is checked on every load, so the copy
# is verifiable rather than merely convenient.
DRAWS = os.path.join(ROOT, 'results', 'e43', 'reserved_draws.json')
DRAWS_SHA = '686ce2e3d03d1e3e87d79cba74810f0f1ad2c945a240fab0bad9328dee114cff'
OUT = os.path.join(ROOT, 'data', 'sc', 'reserved')
SYSTEMS = {'rd': pdes.RD, 'ks': pdes.KS, 'film': pdes.Film,
           'burgers': pdes.Burgers}


def load_reserved(path):
    import hashlib
    raw = open(path, 'rb').read()
    got = hashlib.sha256(raw).hexdigest()
    assert got == DRAWS_SHA, (
        f'draws.json digest {got} != the frozen {DRAWS_SHA}; the reservation '
        'has changed and nothing downstream of it can be trusted')
    rows = json.loads(raw)
    return [r for r in rows if r.get('role') == 'reserved_confirmation']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=list(SYSTEMS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = load_reserved(DRAWS)
    report = {'protocol': 'docs/PROSPECTIVE_PREREGISTRATION.md',
              'draws_sha256': DRAWS_SHA, 'systems': {}}

    for name in a.systems:
        S = SYSTEMS[name]()
        rows = [r for r in res if r['system'] == name]
        assert len(rows) == 64, f'{name}: {len(rows)} reserved, expected 64'
        # the frozen parameter names must match, or the vectors mean nothing
        assert all(set(r['params']) == set(S.param_names) for r in rows), (
            f'{name}: reserved params do not match param_names '
            f'{tuple(S.param_names)}')
        trs, t0 = [], time.time()
        for r in rows:
            try:
                trs.append(S.solve({k: float(v) for k, v in r['params'].items()}))
            except Exception as e:                           # pragma: no cover
                trs.append(None)
                print(f"  {r['id']}: FAILED {type(e).__name__}")
        wall = time.time() - t0
        ok = [t for t in trs if t is not None and t.status == 0]
        d = scd.build(S, trs)
        scd.save(os.path.join(OUT, f'{name}.h5'), d)
        report['systems'][name] = {
            'reserved': len(rows), 'solved': len(ok),
            'non_singular_or_failed': len(rows) - len(ok),
            'wall_s': wall,
            'ids': [r['id'] for r in rows],
        }
        print(f'{name:8s} {len(ok):3d}/{len(rows)} usable  {wall:6.1f}s  '
              f'-> data/sc/reserved/{name}.h5', flush=True)

    json.dump(report, open(os.path.join(OUT, 'generation.json'), 'w'), indent=1)
    print('wrote data/sc/reserved/generation.json  (no scoring performed)')


if __name__ == '__main__':
    main()
