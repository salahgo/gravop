#!/usr/bin/env python3
"""Figures: particle references against the fluid model under three closures.

fig_trajectories: central density (mean inside 0.3% of the mass) of haloes B, C
  and D2, every NSphere run against the fluid model.
fig_profiles: density profiles of haloes C and D1 at three epochs, seed-mean of
  the NSphere runs against the fluid model.
Particle measures: results/independent_halo_20261107/nsphere_*.npz and
experiments/research_extensions/{multifidelity,closure_design}/nsphere/*/measures.npz.
The fluid runs are recomputed with gravop.closure (as in compare_D_structure.py).

    python scripts/fig_closure_structure.py [--out DIR]     (about 1 min)
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXT = os.path.join(ROOT, 'experiments', 'research_extensions')
OUT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(ROOT, 'results', 'figures')
sys.path[:0] = [os.path.join(EXT, 'scripts'), os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]
import independent_halo_analysis as IH  # noqa: E402
from gravop import closure, solver  # noqa: E402

plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.4, 'legend.frameon': False, 'pdf.fonttype': 42})
CL = {'harmonic': ((1.0, 0.712), '#2a78d6', r'harmonic'),
      'fit5': ((0.58, 0.85), '#8a5cc2', r'five-halo fit, $\alpha=0.58$'),
      'fit': ((0.535, 0.886), '#1baf7a', r'seven-halo fit, $\alpha=0.535$')}
PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))['haloes']
PRE.update(json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_E.json')))['haloes'])
for h in ('D1', 'D2', 'E1', 'E2'):
    IH.HALO[h] = dict(mass=PRE[h]['mass'], rs=PRE[h]['rs'], falloff=PRE[h]['falloff'],
                      kappa=PRE[h]['kappa'], tfinal_tdyn=PRE[h]['tfinal_tdyn'])


def particle_runs(h):
    out = []
    if h in ('B', 'C'):
        d = os.path.join(ROOT, 'results', 'independent_halo_20261107')
        out += [np.load(os.path.join(d, f)) for f in sorted(os.listdir(d))
                if f.startswith(f'nsphere_{h}') and f.endswith('.npz')]
        d = os.path.join(EXT, 'multifidelity', 'nsphere')
    else:
        d = os.path.join(EXT, 'closure_design', 'nsphere')
    out += [np.load(os.path.join(d, r, 'measures.npz')) for r in sorted(os.listdir(d))
            if r.startswith(h + '_') and 'dt2' not in r and os.path.exists(os.path.join(d, r, 'measures.npz'))]
    return out


def fluid(h, a, C):
    u = IH.halo_units(h)
    r, m, rho, p = solver.profile_from_rho(u['shape'], r_min=0.01, r_max=85.0, n_shells=300)
    run = closure.run_halo_g(sigma_m=u['sigma_hat'], C=C, alpha=a, r=r, m=m, rho=rho, p=p, eps_t=3e-4,
                             timestep_mode='implicit', rho_c_factor=1e4, kn_min=0.02,
                             t_end=u['t_end'], max_snaps=3000)
    meas = [IH.fluid_measures(run['r'][i], run['m'], run['rho'][i], run['p'][i], u['I'])
            for i in range(len(run['t']))]
    return dict(t=run['t'], mean_rho=np.array([x[1] for x in meas]), prof=np.array([x[3] for x in meas]))


def main():
    HS = ('B', 'C', 'D1', 'D2', 'E1', 'E2')
    F = {(h, k): fluid(h, *CL[k][0]) for h in HS for k in CL}
    P = {h: particle_runs(h) for h in HS}
    print({h: len(v) for h, v in P.items()})

    fig, ax = plt.subplots(3, 2, figsize=(6.4, 6.4))
    ax = ax.ravel()
    for a, h, kn in zip(ax, ('B', 'C', 'D2', 'D1', 'E1', 'E2'), ('6.7', '1.8', '1.1', '0.6', '1.1, c=19', '0.45, c=10')):
        for j, d in enumerate(P[h]):
            a.plot(d['t'], d['mean_rho'][:, 1] / d['mean_rho'][0, 1], color='#9a9a9a', lw=0.7,
                   label=f'NSphere ({len(P[h])} runs)' if j == 0 else None)
        tmax = 0
        for k, (_, col, lab) in CL.items():
            f = F[(h, k)]
            a.plot(f['t'], f['mean_rho'][:, 1] / f['mean_rho'][0, 1], color=col, label=lab)
        tmax = max(d['t'][np.argmax(d['mean_rho'][:, 1] / d['mean_rho'][0, 1] > 30)] for d in P[h]
                   if (d['mean_rho'][:, 1] / d['mean_rho'][0, 1] > 30).any())
        a.set_yscale('log'); a.set_ylim(0.03, 30); a.set_xlim(0, 1.1 * tmax)
        if h in ('E1', 'E2'):
            a.set_xlabel(r'$t/t_0$')
        if h in ('B', 'D2', 'E1'):
            a.set_ylabel('central density / initial')
        a.set_title(fr'halo {h} ($\mathrm{{Kn}}_c={kn}$)', loc='left', fontsize=7.5)
    ax[0].legend(fontsize=5.8, loc='upper left')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_trajectories.{ext}'), dpi=200)

    rc = np.sqrt(IH.RBINS[1:] * IH.RBINS[:-1])
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.4), sharex=True)
    for row, h in enumerate(('C', 'D1')):
        tx = np.mean([IH.collapse_markers(d['t'], d['mean_rho'][:, 1])['t_x10'] for d in P[h]])
        for col, frac in enumerate((0.3, 0.6, 0.9)):
            a = ax[row, col]
            t = frac * tx
            pp = np.array([IH.interp_at(d['t'], d['prof'], t) for d in P[h]])
            pp[~np.isfinite(pp)] = np.nan
            ok = rc < 0.6 * IH.HALO[h]['falloff']
            a.plot(rc[ok], np.nanmean(pp, 0)[ok], 'o', ms=2.5, color='#555555', label='NSphere (seed mean)')
            for k, (_, c, lab) in CL.items():
                a.plot(rc[ok], IH.interp_at(F[(h, k)]['t'], F[(h, k)]['prof'], t)[ok], color=c, label=lab)
            a.set_xscale('log')
            a.set_title(fr'halo {h}, $t = {frac:.1f}\,t_{{x10}}$', loc='left', fontsize=7.5)
            if row == 1:
                a.set_xlabel(r'$r/r_s$')
        ax[row, 0].set_ylabel(r'$\log_{10}\rho/\rho_s$')
    ax[0, 0].legend(fontsize=5.8, loc='lower left')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_profiles.{ext}'), dpi=200)


if __name__ == '__main__':
    main()
