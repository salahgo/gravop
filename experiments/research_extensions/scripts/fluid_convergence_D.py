#!/usr/bin/env python3
"""Fluid-side convergence of the D1/D2 predictions (shells x2, eps_t /3), harmonic and literature."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
import independent_halo_analysis as IH
from gravop import closure, solver
PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))
out = {}
for h in ('D1', 'D2'):
    d = PRE['haloes'][h]
    IH.HALO[h] = dict(mass=d['mass'], rs=d['rs'], falloff=d['falloff'], kappa=d['kappa'], tfinal_tdyn=1.0)
    u = IH.halo_units(h)
    for name, (a, C) in (('harmonic', (1.0, 0.712)), ('literature', (0.5, 0.9))):
        for ns, eps in ((300, 3e-4), (600, 3e-4), (300, 1e-4)):
            r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=ns)
            run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=a, r=r, m=m, rho=rho, p=p, eps_t=eps,
                                     timestep_mode='implicit', rho_c_factor=1e4, kn_min=0.02, t_end=400.0, max_snaps=3000)
            mr = np.array([IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I'])[1][1] for i in range(len(run['t']))])
            t = IH.collapse_markers(run['t'], mr)['t_x10'] * u['t0_myr'] / u['tdyn_myr']
            out[f'{h}|{name}|{ns}|{eps}'] = t
            print(h, name, ns, eps, round(t, 2), flush=True)
json.dump(out, open(os.path.join(EXT, 'closure_design', 'fluid_convergence_D.json'), 'w'), indent=1)
