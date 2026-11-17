"""E9 zoom grid -- the solver-grid posterior that the emulator posterior is
compared with, and the comparison itself.

`e9_inference.py grid` scans the whole prior on a 31x31 grid, which aliases the
thin curved likelihood ridge of the tau = 0.7 mock. The posterior quoted in the
manuscript is a finer grid over a sub-box. This script is its producer:

  original  91 x 91 nodes on log10 sigma0 in [-0.1, 0.8], log10 w in [-0.6, 1.2]
            -> results/e9/grid_zoom.npz        (reproduces the stored file)
  extended  111 x 106 nodes on [-0.1, 1.0] x [-0.6, 1.5], the same node spacing
            extended to the prior's upper edges, so that no emulator sample and
            no posterior mass along the ridge falls outside the grid
            -> results/e9/grid_zoom_extended.npz

Each node is one direct solver evaluation with e9_inference.solver_profile_at
and the Gaussian likelihood of e9_inference's grid stage, so a node's value is
identical whichever grid it belongs to; `--reuse` takes the nodes an existing
grid already holds instead of solving them again.

`summary` compares a grid with the emulator chain (results/e9/samples.npy):
grid and emulator means and standard deviations, and the 2-D overlap
coefficient sum_cells min(P_grid, P_emulator) with the emulator histogrammed on
the grid's cells. Two normalisations are reported: over the samples inside the
grid (the convention of the stored summary_zoom.json) and over all samples.

Usage:
  e9_zoom_grid.py solve --box original --workers 20
  e9_zoom_grid.py solve --box extended --reuse results/e9/grid_zoom.npz --workers 6
  e9_zoom_grid.py summary --box original     # -> summary_zoom.json values
  e9_zoom_grid.py summary --box extended     # -> summary_zoom_extended.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e9_inference as E9  # noqa: E402

OUT = E9.OUT
BOXES = {
    'original': dict(lsig=(-0.1, 0.8, 91), lw=(-0.6, 1.2, 91),
                     grid='grid_zoom.npz', summary='summary_zoom.json'),
    'extended': dict(lsig=(-0.1, 1.0, 111), lw=(-0.6, 1.5, 106),
                     grid='grid_zoom_extended.npz',
                     summary='summary_zoom_extended.json'),
}


def nodes(box):
    b = BOXES[box]
    return np.linspace(*b['lsig']), np.linspace(*b['lw'])


def log_like(lrho, lrho_obs):
    if lrho is None:
        return -np.inf
    return -0.5 * float(np.sum((np.asarray(lrho) - lrho_obs) ** 2
                               / E9.NOISE_DEX ** 2))


def solve(box, workers, reuse):
    mock = json.load(open(os.path.join(OUT, 'mock.json')))
    lrho_obs = np.array(mock['lrho_obs'])
    g1, g2 = nodes(box)
    logpost = np.full((len(g1), len(g2)), np.nan)
    if reuse:
        old = np.load(reuse)
        for i, a in enumerate(g1):
            ii = np.flatnonzero(np.abs(old['lsig'] - a) < 1e-9)
            if not len(ii):
                continue
            for j, b in enumerate(g2):
                jj = np.flatnonzero(np.abs(old['lw'] - b) < 1e-9)
                if len(jj):
                    logpost[i, j] = old['logpost'][ii[0], jj[0]]
    todo = [(i, j) for i in range(len(g1)) for j in range(len(g2))
            if np.isnan(logpost[i, j])]
    print(f'{box}: {logpost.size} nodes, {logpost.size - len(todo)} reused, '
          f'{len(todo)} to solve on {workers} workers', flush=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(E9.grid_worker,
                        [(g1[i], g2[j], mock['t_obs']) for i, j in todo],
                        chunksize=4)
    for (i, j), (_, _, lrho, _) in zip(todo, rows):
        logpost[i, j] = log_like(lrho, lrho_obs)
    wall = time.time() - t0
    np.savez(os.path.join(OUT, BOXES[box]['grid']), lsig=g1, lw=g2,
             logpost=logpost, wall=wall, n_solved=len(todo),
             workers=workers)
    print(f'wrote {BOXES[box]["grid"]} in {wall:.0f} s', flush=True)


def summarise(box):
    g = np.load(os.path.join(OUT, BOXES[box]['grid']))
    x, y, lp = g['lsig'], g['lw'], g['logpost']
    P = np.exp(np.where(np.isfinite(lp), lp - np.max(lp[np.isfinite(lp)]),
                        -np.inf))
    P /= P.sum()
    chain = np.load(os.path.join(OUT, 'samples.npy'))
    dx, dy = x[1] - x[0], y[1] - y[0]
    H, _, _ = np.histogram2d(chain[:, 0], chain[:, 1],
                             bins=[np.r_[x - dx / 2, x[-1] + dx / 2],
                                   np.r_[y - dy / 2, y[-1] + dy / 2]])
    mx, my = P.sum(1) @ x, P.sum(0) @ y
    sx = np.sqrt(P.sum(1) @ (x - mx) ** 2)
    sy = np.sqrt(P.sum(0) @ (y - my) ** 2)
    edge = P[0].sum() + P[-1].sum() + P[:, 0].sum() + P[:, -1].sum()
    imax = np.unravel_index(np.argmax(P), P.shape)
    out = {
        'box': {'lsig': [float(x[0]), float(x[-1]), len(x)],
                'lw': [float(y[0]), float(y[-1]), len(y)]},
        'zoom_grid_mean': [float(mx), float(my)],
        'zoom_grid_std': [float(sx), float(sy)],
        'grid_mode': [float(x[imax[0]]), float(y[imax[1]])],
        'grid_mass_on_boundary_cells': float(edge),
        'emulator_mean': chain.mean(0).tolist(),
        'emulator_std': chain.std(0).tolist(),
        'overlap_2d': float(np.minimum(P, H / H.sum()).sum()),
        'overlap_2d_all_samples': float(np.minimum(P, H / len(chain)).sum()),
        'frac_samples_in_box': float(H.sum() / len(chain)),
        'offset_in_grid_sigma': [float((chain[:, 0].mean() - mx) / sx),
                                 float((chain[:, 1].mean() - my) / sy)],
        'emulator_over_grid_std': [float(chain[:, 0].std() / sx),
                                   float(chain[:, 1].std() / sy)],
        'truth': [float(np.log10(E9.TRUTH['sigma0'])),
                  float(np.log10(E9.TRUTH['w']))],
        'overlap_definition': ('sum over grid cells of min(P_grid, P_emulator), '
                               'emulator histogrammed on the grid cells; '
                               'overlap_2d normalises the histogram over the '
                               'samples inside the grid, overlap_2d_all_samples '
                               'over the whole chain'),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['solve', 'summary'])
    ap.add_argument('--box', choices=sorted(BOXES), default='original')
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--reuse', default=None,
                    help='an existing grid whose coincident nodes are reused')
    a = ap.parse_args()
    if a.stage == 'solve':
        solve(a.box, a.workers, a.reuse)
        return
    out = summarise(a.box)
    if a.box == 'original':
        # the stored record predates this script; check it rather than
        # rewrite it, and fail loudly if this producer does not reproduce it
        ref = json.load(open(os.path.join(OUT, BOXES['original']['summary'])))
        for k in ('zoom_grid_mean', 'zoom_grid_std', 'emulator_mean',
                  'emulator_std', 'overlap_2d', 'frac_samples_in_box'):
            if not np.allclose(ref[k], out[k], rtol=1e-12, atol=0):
                raise SystemExit(f'{k}: stored {ref[k]} != recomputed {out[k]}')
        print('summary_zoom.json reproduced; additional fields:')
    else:
        json.dump(out, open(os.path.join(OUT, BOXES[a.box]['summary']), 'w'),
                  indent=1)
        print(f'wrote {BOXES[a.box]["summary"]}')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
