#!/usr/bin/env python3
"""The (alpha, C) grid of closure_scan.py on the design haloes D1, D2 (t_x10 in t_dyn)."""
import json, os, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import closure_scan as CS
import family_posterior as FP


def job(a):
    h, al, C = a
    import gfamilies as F
    gx, gf = F.alpha(al)
    return f'{h}|{al}|{C}', FP.t_x10(h, gx, gf, float(C))


if __name__ == '__main__':
    tasks = [(h, al, C) for h in ('D1', 'D2') for al in CS.ALPHAS for C in CS.CS]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as p:
        res = dict(p.map(job, tasks, chunksize=1))
    json.dump(res, open(os.path.join(EXT, 'learned_closures', 'closure_scan_D.json'), 'w'), indent=1)
    print('done', len(res))
