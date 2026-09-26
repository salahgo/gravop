"""E1 -- cache the halo104 merger-tree trajectories for the E30 sample.

Builds the analysis-ready time series that every later experiment (E2-E7) reads,
so the 5 GB tree files are traversed ONCE.

Sample: exactly E30's 81 haloes, so the trajectory work inherits a selection that
is already published, already gated, and already reasoned about -- rather than
inventing a second sample nobody has audited.

Matching: E30's pairs are keyed by the parametric catalogue's `vd100id`, which
E30 established does NOT cross-match Rockstar `id`. A PHYSICAL match does, and
exactly: on (vmax, Mvir*h) for CDM and (vmax, RVmax*h) for GroupSIDM every one of
the 81 finds a unique host at <= 2.2e-16 dex. The `*h` on both mass and radius is
the same h = 0.7 convention DATA.md records; getting it wrong drops the match
rate to zero, which is how it was verified rather than assumed.

Stored per halo, both runs, oldest-first: scale, Mvir, Rvir, rs, vmax, RVmax,
vrms, pid, scale_of_last_MM, num_prog. Arrays are padded to 236 with NaN and
accompanied by a length, because branches differ in length.

Output: results/e31/trajectories.npz
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import concerto_trees as ct          # noqa: E402
import e30_concerto as e30           # noqa: E402

H = 0.7
NMAX = 236
COLS = {'scale': 0, 'Mvir': 10, 'Rvir': 11, 'rs': 12, 'vrms': 13,
        'scale_of_last_MM': 15, 'vmax': 16, 'pid': 5, 'num_prog': 4,
        'RVmax': 60}
TREES = {'cdm': os.path.join(ROOT, 'data/d3_concerto/Halo104_trees/cdm_trees'),
         'sidm': os.path.join(ROOT, 'data/d3_concerto/Halo104_trees/GroupSIDM_trees')}
TARS = {'cdm': 'LMC_Halo104_CDM_RCT.tar',
        'sidm': 'LMC_Halo104_GroupSIDM_RCT.tar'}


def hlist(tar):
    """z = 0 Rockstar catalogue as (id, Mvir, vmax, RVmax) plus the raw rows."""
    r = subprocess.run(['tar', 'xOf', os.path.join(ROOT, 'data/d3_concerto', tar)],
                       capture_output=True, text=True)
    ids, mv, vm, rv, raw = [], [], [], [], []
    for line in r.stdout.splitlines():
        if line.startswith('#'):
            continue
        f = line.split()
        if len(f) < 61:
            continue
        ids.append(int(f[1])); mv.append(float(f[10]))
        vm.append(float(f[16])); rv.append(float(f[60])); raw.append(f)
    return (np.array(ids), np.array(mv), np.array(vm), np.array(rv), raw)


def exact_match(x1, x2, y1, y2, label):
    """Nearest neighbour in log-log; assert it is exact and unique."""
    d = np.sqrt((np.log10(x1[:, None]) - np.log10(x2[None, :])) ** 2
                + (np.log10(y1[:, None]) - np.log10(y2[None, :])) ** 2)
    j = d.argmin(1)
    dm = d[np.arange(len(x1)), j]
    assert dm.max() < 1e-9, f'{label}: match not exact, max {dm.max():.2e} dex'
    assert len(set(j.tolist())) == len(j), f'{label}: match not unique'
    return j


def main():
    p = e30.load_pairs()
    keep = np.isfinite(p['cdm_vmax']) & np.isfinite(p['sidm_vmax'])
    keep &= (p['cdm_vmax'] > 8.0) & (p['cdm_rmax'] > 0)
    keep &= (p['sidm_vmax'] > 0) & (p['sidm_rmax'] > 0) & (p['pid'] == -1)
    sel = np.nonzero(keep)[0]
    print(f'E30 sample: {len(sel)} haloes', flush=True)

    out = {}
    for run in ('cdm', 'sidm'):
        ids, hm, hv, hr, _ = hlist(TARS[run])
        if run == 'cdm':
            j = exact_match(p['cdm_vmax'][sel], hv, p['mcdm'][sel] * H, hm, 'cdm')
        else:
            j = exact_match(p['sidm_vmax'][sel], hv, p['sidm_rmax'][sel] * H, hr, 'sidm')
        root = ids[j]
        print(f'{run}: matched {len(root)} unique roots', flush=True)

        d = TREES[run]
        f = os.path.join(d, 'tree_0_0_0.dat')
        size = os.path.getsize(f)
        loc = ct.load_locations(os.path.join(d, 'locations.dat'))

        arr = {k: np.full((len(root), NMAX), np.nan) for k in COLS}
        n = np.zeros(len(root), dtype=int)
        # read in file order so the disk is swept once, not seeked randomly
        order = np.argsort([loc[int(r)] for r in root])
        for c, oi in enumerate(order):
            s, e = ct.tree_span(loc, int(root[oi]), size)
            mb = ct.main_branch(ct.read_tree_block(f, s, e))
            mb = mb[::-1]                      # oldest first
            n[oi] = len(mb)
            for k, ci in COLS.items():
                arr[k][oi, :len(mb)] = [float(r[ci]) for r in mb]
            if (c + 1) % 20 == 0:
                print(f'  {run} {c+1}/{len(root)}', flush=True)
        out[f'{run}_root_id'] = root
        out[f'{run}_n'] = n
        for k in COLS:
            out[f'{run}_{k}'] = arr[k]

    out['cdm_vmax_z0'] = p['cdm_vmax'][sel]
    out['sidm_vmax_z0'] = p['sidm_vmax'][sel]
    out['cdm_rmax_z0'] = p['cdm_rmax'][sel]
    out['sidm_rmax_z0'] = p['sidm_rmax'][sel]
    out['mcdm'] = p['mcdm'][sel]
    out['tr1_theirs'] = p['tr1'][sel]

    dst = os.path.join(ROOT, 'results', 'e31')
    os.makedirs(dst, exist_ok=True)
    np.savez_compressed(os.path.join(dst, 'trajectories.npz'), **out)
    print('wrote', os.path.join(dst, 'trajectories.npz'))
    print('branch length: cdm median %.0f, sidm median %.0f'
          % (np.median(out['cdm_n']), np.median(out['sidm_n'])))


if __name__ == '__main__':
    sys.exit(main())
