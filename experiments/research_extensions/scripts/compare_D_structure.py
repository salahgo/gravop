#!/usr/bin/env python3
"""D haloes: full-trajectory comparison of particles and fluid closures.

Per closure: RMS of ln(rho_c) along the evolution on the particle time grid up to
0.9 t_x10(particles); median |log10| Lagrangian-radius and dispersion errors;
density-profile RMS (dex) before collapse (bins with >= 100 particles inside
0.6 c r_s), all with the same measures as haloes B and C.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
import independent_halo_analysis as IH
import nsphere_ext as NX
from gravop import closure, solver
PINNED = {'D1': ['D1_s21', 'D1_s22'], 'D2': ['D2_s21', 'D2_s22']}
PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))


def fluid(h, a, C):
    d = PRE['haloes'][h]
    IH.HALO[h] = dict(mass=d['mass'], rs=d['rs'], falloff=d['falloff'], kappa=d['kappa'], tfinal_tdyn=1.0)
    u = IH.halo_units(h)
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=a, r=r, m=m, rho=rho, p=p, eps_t=3e-4,
                             timestep_mode='implicit', rho_c_factor=1e4, kn_min=0.02, t_end=400.0, max_snaps=3000)
    meas = [IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I']) for i in range(len(run['t']))]
    return u, dict(t=run['t'], lagr_r=np.array([x[0] for x in meas]), mean_rho=np.array([x[1] for x in meas]),
                   disp=np.array([x[2] for x in meas]), prof=np.array([x[3] for x in meas]))


def main(h):
    # Pinned run list: the archived structure records used the first two 1e5
    # seeds, the only runs finished when they were computed. Pass --all to use
    # every finished run except the half-time-step control.
    runs = PINNED[h] if '--all' not in sys.argv else [
        r for r, c in NX.RUNS.items() if c.get('halo') == h
        and os.path.exists(os.path.join(NX.run_dir(r), 'measures.npz'))
        and not r.split('_')[1].startswith('dt2')]
    P = [np.load(os.path.join(NX.run_dir(r), 'measures.npz')) for r in runs]
    out = {'runs': runs, 'closures': {}}
    vol = 4 / 3 * np.pi * (IH.RBINS[1:] ** 3 - IH.RBINS[:-1] ** 3)
    rc = np.sqrt(IH.RBINS[1:] * IH.RBINS[:-1])
    for name, (a, C) in (('harmonic', (1.0, 0.712)), ('literature', (0.5, 0.9)), ('alpha_A', (0.5, 0.878)),
                         ('valley_a0.56', (0.557, 0.847)), ('valley_best', (0.669, 0.779)), ('valley_a1.5', (1.5, 0.703))):
        u, f = fluid(h, a, C)
        e_rho, e_r, e_s, e_p = [], [], [], []
        for d in P:
            tx = IH.collapse_markers(d['t'], d['mean_rho'][:, 1])['t_x10']
            for k, tt in enumerate(d['t']):
                if 0 < tt < 0.9 * tx and tt <= f['t'][-1]:
                    e_rho.append(np.log(IH.interp_at(f['t'], f['mean_rho'], tt)[1] / d['mean_rho'][k][1]))
                    e_r.append(np.abs(np.log10(IH.interp_at(f['t'], f['lagr_r'], tt)[1:] / d['lagr_r'][k][1:])))
                    e_s.append(np.abs(IH.interp_at(f['t'], f['disp'], tt)[1:] / d['disp'][k][1:] - 1))
                    if tt < 0.8 * tx:
                        mp = IH.interp_at(f['t'], f['prof'], tt)
                        cnt = 10 ** d['prof'][k] * (vol / (4 * np.pi)) / (u['I'] / int(d['N']))
                        ok = np.isfinite(d['prof'][k]) & (cnt >= 100) & (rc < 0.6 * PRE['haloes'][h]['falloff'])
                        e_p.append(np.sqrt(np.mean((mp[ok] - d['prof'][k][ok]) ** 2)))
        out['closures'][name] = dict(central_density_rms_ln=float(np.sqrt(np.mean(np.square(e_rho)))),
                                     central_density_bias_ln=float(np.mean(e_rho)),
                                     lagr_radius_dex_median=float(np.median(e_r)),
                                     dispersion_rel_median=float(np.median(e_s)),
                                     profile_rms_dex_median=float(np.median(e_p)))
        print(h, name, {k: round(v, 4) for k, v in out['closures'][name].items()}, flush=True)
    json.dump(out, open(os.path.join(EXT, 'closure_design', f'structure_{h}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
