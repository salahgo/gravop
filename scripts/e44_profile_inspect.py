#!/usr/bin/env python3
"""E44 -- hand-inspect the profiles at the smallest R_Vmax, as the suite's
authors asked.

The Concerto authors' advice on convergence was adopted in P49: R_Vmax is well
measured above ~2000 particles, and a small R_Vmax is not itself a problem
because it means many particles at small radii. They attached one caution to
that, and it is the only part this project has not acted on:

    "Of course, if R_Vmax is extremely small, you should be cautious and
     manually inspect the density and velocity dispersion profiles."

The smallest R_Vmax in our matched sample sits at 2.05 softening lengths. This
builds its density and velocity-dispersion profiles from the z=0 particles and
looks at them, rather than continuing to assert that the catalogue summary is
trustworthy there.

The halo: SIDM id 14139719, Mvir 5.20e7 Msun, 8249 particles, R_Vmax 0.082
kpc/h against a softening of 0.04, and a measured R_Vmax shrinkage of -0.669 dex
against its CDM counterpart. It is one of the 17 contracted haloes that carry
the paper's structural result, so if any halo's profile matters, it is this one.

Reads the 16 Gadget-1 binary chunks, keeping only the high-resolution particle
type (type 1, m = 6294 Msun), and filters per chunk so the full 44 million
positions are never resident at once.

Output: results/e44/profile_inspection.json
"""
import json
import os
import struct
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'e44')
TAR = os.path.join(ROOT, 'data', 'd3_concerto',
                   'LMC_Halo104_GroupSIDM_particles.tar')
SNAPDIR = ('central/groups/carnegie_poc/enadler/zoomins/Halo104/'
           'output_GroupSIDM_32K')
H, EPS_KPCH = 0.7, 0.04          # softening, comoving kpc/h
HIRES_TYPE = 1


def read_chunk(path):
    """Positions (Mpc/h) and velocities (km/s) of the high-resolution type."""
    with open(path, 'rb') as f:
        assert struct.unpack('i', f.read(4))[0] == 256
        h = f.read(256)
        f.read(4)
        npart = np.frombuffer(h[0:24], dtype=np.int32)
        n = int(npart.sum())
        f.read(4)
        pos = np.frombuffer(f.read(12 * n), dtype=np.float32).reshape(n, 3)
        f.read(8)
        vel = np.frombuffer(f.read(12 * n), dtype=np.float32).reshape(n, 3)
    lo = int(npart[:HIRES_TYPE].sum())
    hi = lo + int(npart[HIRES_TYPE])
    return pos[lo:hi], vel[lo:hi]


def main():
    cx, cy, cz, rvir, rvmax, mvir = np.load('/tmp/halo_target.npy')
    centre = np.array([cx, cy, cz])
    rmax_mpc = 3.0 * rvir / 1000.0          # keep 3 Rvir, in Mpc/h
    m_part = 4.40595587e-07 * 1e10 / H      # Msun

    keepP, keepV = [], []
    for i in range(16):
        rel = f'{SNAPDIR}/snapshot_235.{i}'
        subprocess.run(['tar', 'xf', TAR, '-C', '/tmp/snap', rel], check=True)
        p, v = read_chunk(os.path.join('/tmp/snap', rel))
        d = p - centre
        sel = np.all(np.abs(d) < rmax_mpc, axis=1)
        if sel.any():
            keepP.append(d[sel]); keepV.append(v[sel])
        os.remove(os.path.join('/tmp/snap', rel))
        print(f'  chunk {i:2d}: {int(sel.sum()):7d} kept', flush=True)
    d = np.concatenate(keepP) * 1000.0 / H      # physical kpc (a = 1)
    v = np.concatenate(keepV)
    r = np.linalg.norm(d, axis=1)
    inside = r < 3.0 * rvir / H
    d, v, r = d[inside], v[inside], r[inside]
    # recentre on the densest region: the catalogue centre is a shrinking-sphere
    # result and a few kpc of offset would flatten the inner profile spuriously
    for frac in (1.0, 0.5, 0.25, 0.1):
        m = r < frac * rvir / H
        if m.sum() < 50:
            break
        c = d[m].mean(0)
        d = d - c
        r = np.linalg.norm(d, axis=1)
    vbulk = v[r < 0.5 * rvir / H].mean(0)

    eps_phys = EPS_KPCH / H
    edges = np.logspace(np.log10(0.2 * eps_phys), np.log10(3.0 * rvir / H), 25)
    mid = np.sqrt(edges[1:] * edges[:-1])
    cnt, _ = np.histogram(r, bins=edges)
    vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = cnt * m_part / vol
    sig = np.full(len(mid), np.nan)
    for k in range(len(mid)):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if m.sum() >= 10:
            sig[k] = float(np.sqrt(((v[m] - vbulk) ** 2).sum(1).mean() / 3.0))

    rows = [{'r_kpc': float(mid[k]), 'r_over_eps': float(mid[k] / eps_phys),
             'n_particles': int(cnt[k]),
             'rho_Msun_kpc3': float(rho[k]) if cnt[k] else None,
             'sigma_1d_kms': None if np.isnan(sig[k]) else float(sig[k])}
            for k in range(len(mid))]
    ok = cnt > 0
    lr, lrho = np.log10(mid[ok]), np.log10(np.maximum(rho[ok], 1e-30))
    inner = (mid[ok] > 0.5 * eps_phys) & (mid[ok] < 4 * eps_phys)
    slope = (float(np.polyfit(lr[inner], lrho[inner], 1)[0])
             if inner.sum() >= 3 else None)

    out = {
        'why': "the Concerto authors' one retained caution: at extremely small "
               'R_Vmax, inspect the density and dispersion profiles by hand',
        'halo': {'sidm_id': 14139719, 'Mvir_Msun': float(mvir / H),
                 'n_particles_virial': int(round(mvir / H / m_part)),
                 'Rvir_kpc': float(rvir / H),
                 'R_Vmax_kpc': float(rvmax / H),
                 'R_Vmax_over_eps': float(rvmax / EPS_KPCH),
                 'softening_kpc': float(eps_phys)},
        'm_particle_Msun': float(m_part),
        'n_particles_within_3Rvir': int(len(r)),
        'n_within_R_Vmax': int((r < rvmax / H).sum()),
        'n_within_one_softening': int((r < eps_phys).sum()),
        'inner_log_slope_0p5_to_4_eps': slope,
        'profile': rows,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'profile_inspection.json'), 'w'),
              indent=1)
    print(f"\nparticles within 3 Rvir: {len(r)}, within R_Vmax: "
          f"{out['n_within_R_Vmax']}, within one softening: "
          f"{out['n_within_one_softening']}")
    print(f"inner log slope (0.5-4 eps): {slope}")
    print(f"\n{'r/eps':>8} {'n':>7} {'rho':>12} {'sigma_1d':>9}")
    for row in rows:
        if row['n_particles']:
            print(f"{row['r_over_eps']:8.2f} {row['n_particles']:7d} "
                  f"{row['rho_Msun_kpc3']:12.4e} "
                  f"{row['sigma_1d_kms'] if row['sigma_1d_kms'] else float('nan'):9.3f}")


if __name__ == '__main__':
    main()
