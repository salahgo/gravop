#!/usr/bin/env python3
"""Experiment G: is the closure sensitivity a universal function of the core
Knudsen number?

Hypothesis (from the design scan, constant cross-sections, c = 6..30): the
collapse-time ratio between closures, R = ln(t_lit / t_harm), depends on the
halo only through Kn_c, the central Knudsen number at the largest core. Test on
haloes the hypothesis was not built from: velocity-dependent (Yukawa, Tulin-Yu
transfer cross-section) haloes over a range of w, plus NFW without truncation.
Prediction: a smooth curve fitted to the constant-cross-section design scan
(R vs ln Kn_c, cubic spline) predicts R for the new haloes to within a few
percent. Falsified if residuals are comparable to R itself.
"""
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUT = os.path.join(EXT, 'physics_discovery')
CL = {'harmonic': (1.0, 0.712), 'literature': (0.5, 0.9)}


def job(args):
    sig0, w, c, name = args
    from gravop import closure, solver
    a, C = CL[name]
    if c is None:
        r, m, rho, p = solver.nfw_profile(0.01, 100.0, 300)
    else:
        shape = lambda x: 1.0 / ((x + 0.01) * (1 + x) ** 2 * (1 + (x / c) ** 10))
        r, m, rho, p = solver.profile_from_rho(shape, r_min=0.01, r_max=85.0, n_shells=300)
    run = closure.run_halo_g(sigma_m=sig0, w=w, model_lmfp='yukawa_tu', model_smfp='yukawa_tu', C=C, alpha=a,
                             r=r, m=m, rho=rho, p=p, eps_t=3e-4, timestep_mode='implicit',
                             rho_c_factor=1e4, kn_min=0.02, t_end=6000.0, max_snaps=3000)
    import independent_halo_analysis as IH
    I = float(run['m'][-1])
    mr = np.array([IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], I)[1][1]
                   for i in range(len(run['t']))])
    t10 = IH.collapse_markers(run['t'], mr)['t_x10']
    i0 = int(np.argmin(run['rho'][:, 3]))
    from gravop.solver import _F_elastic, model_params
    ml, mpl = model_params('yukawa_tu', 0.0, 5)
    v0 = np.sqrt(run['p'][i0, 3] / run['rho'][i0, 3])  # central dispersion at the largest core
    Fl = _F_elastic(v0 / w, ml, mpl)
    Kn = float(1.0 / (np.sqrt(run['p'][i0, 3]) * sig0 * Fl))
    return dict(sigma0=sig0, w=w, falloff=c, closure=name, t_x10=t10, Kn_c=Kn, status=run['status'])


def main():
    tasks = []
    for w in (0.3, 0.6, 1.2, 2.5):
        for s in np.logspace(np.log10(0.3), np.log10(40), 9):
            for c in (19.0, None):
                for n in CL:
                    tasks.append((float(s), w, c, n))
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        res = pool.map(job, tasks, chunksize=1)
    rev = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    json.dump(dict(meta=dict(git=rev, wall_s=time.time() - t0, note=__doc__), results=res),
              open(os.path.join(OUT, 'universality_vdep.json'), 'w'), indent=1)
    print('done', len(res), time.time() - t0)


if __name__ == '__main__':
    main()
