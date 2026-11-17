#!/usr/bin/env python3
"""Independent haloes B and C against the kinetic NSphere reference.

(a), (b) central density (mean inside the innermost 0.3 per cent of the mass)
for haloes C and B: NSphere runs (seeds; half the time step for C; twice the
particles for both), and the fluid model with the harmonic closure (harmonic,
C = 0.712 calibrated on halo A) and the literature one (alpha = 1/2,
C = 0.9 of van den Bosch & Tokayer, fitted to nothing here).
(c) Transfer diagram: collapse-time error of each closure against the central
Knudsen number at the largest core, for halo A (N-body), B and C (NSphere at
the highest particle number), with the reference uncertainty (seed scatter and
particle-number shift in quadrature) as error bars
(results/closure_20261108/transfer.json).
(d) the production emulator against the solver for both haloes, on their
nearest in-family (untruncated NFW) initial profile.
Reads results/independent_halo_20261107/*.npz; writes
results/figures/fig25_independent_haloes.{pdf,eps,png}.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as FS  # noqa: E402
import independent_halo_analysis as A  # noqa: E402
import json  # noqa: E402

FIG = os.path.join(HERE, '..', 'results', 'figures')
O = A.OUT
CL = os.path.join(HERE, '..', 'results', 'closure_20261108')


def load(name):
    p = os.path.join(O, name)
    return np.load(p) if os.path.exists(p) else None


def main():
    FS.use('plot')
    fig, ax = plt.subplots(2, 2, figsize=(4.9, 3.9))
    a, b, c, d = ax.ravel()
    style = {'C1': ('-', FS.MUTED, 'NSphere'), 'C2': ('-', FS.MUTED, None),
             'C3': ('--', FS.RULE, r'NSphere, $\Delta t/2$'), 'C4': (':', FS.RULE, r'NSphere, $2N$'),
             'B1': ('-', FS.MUTED, None), 'B2': ('-', FS.MUTED, None),
             'B5': (':', FS.RULE, None), 'B6': (':', FS.RULE, None)}
    for axx, h, runs in ((a, 'C', ('C1', 'C2', 'C3', 'C4')), (b, 'B', ('B1', 'B2', 'B5', 'B6'))):
        t0 = A.halo_units(h)['t0_myr'] / 1e3
        for r in runs:
            n = load(f'nsphere_{r}.npz')
            if n is None:
                continue
            ls, col, lab = style[r]
            axx.plot(n['t'] * t0, n['mean_rho'][:, 1], ls=ls, color=col, lw=0.8, label=lab)
        for f, ls, col, lab in ((os.path.join(O, f'solver_{h}_a1.0.npz'), '-', FS.STATE_HUE, 'fluid, harmonic'),
                                (os.path.join(CL, f'fluid_{h}_alpha_lit.npz'), '-.', FS.TAU_HUE,
                                 'fluid, literature')):
            s = np.load(f)
            axx.plot(s['t'] * t0, s['mean_rho'][:, 1], ls=ls, color=col, lw=1.1, label=lab)
        axx.set_yscale('log')
        axx.set_xlabel('t [Gyr]')
        axx.set_ylabel(r'$\bar\rho_c/\rho_s$')
        axx.text(0.03, 0.95, f"({'a' if h == 'C' else 'b'}) halo {h}", transform=axx.transAxes,
                 va='top', fontsize=FS.MINPT)
    a.set_ylim(0.8, 3e4)
    b.set_ylim(0.8, 3e4)
    a.legend(loc='center left', fontsize=FS.MINPT - 0.5, frameon=False, handlelength=1.6)
    T = json.load(open(os.path.join(CL, 'transfer.json')))['haloes']
    ms = {'harmonic': ('o', FS.STATE_HUE, 'harmonic (fit on A)'),
          'alpha_A': ('^', FS.MUTED, r'$\alpha=1/2$ (fit on A)'),
          'alpha_lit': ('s', FS.TAU_HUE, 'literature (no fit)')}
    for k, (mk, col, lab) in ms.items():
        xs, ys, es = [], [], []
        for h in ('A', 'B', 'C'):
            H = T[h]
            xs.append(H['harmonic']['Kn_c_at_largest_core'])
            if h == 'A':
                ys.append(100 * H[k]['clock_error'])
                es.append(0.0)
                continue
            ref = 'N2e5_mean' if H['references'].get('N2e5_mean') else 'N1e5_seed_mean'
            ys.append(100 * H[k][f'clock_error_vs_{ref}'])
            lo, hi = H['references']['N1e5_seed_mean'], H['references'].get('N2e5_mean')
            sp = H.get('N2e5_halfspread_rel') if hi and H.get('N2e5_halfspread_rel') else H['N1e5_seed_halfspread_rel']
            res = abs(hi / lo - 1) if hi else 0.0
            es.append(100 * float(np.hypot(sp or 0.0, res)))
        off = {'harmonic': 0.97, 'alpha_A': 1.0, 'alpha_lit': 1.03}[k]
        c.errorbar(np.array(xs) * off, ys, yerr=es, fmt=mk, ms=3.6, color=col, mfc='none' if k != 'alpha_lit' else col,
                   mec=col, elinewidth=0.7, capsize=1.5, label=lab)
    for h, x in (('A', T['A']['harmonic']['Kn_c_at_largest_core']), ('B', T['B']['harmonic']['Kn_c_at_largest_core']),
                 ('C', T['C']['harmonic']['Kn_c_at_largest_core'])):
        c.text(x, -14, h, ha='center', fontsize=FS.MINPT, color=FS.INK)
    c.axhline(0, color=FS.RULE, lw=0.6)
    c.set_xscale('log')
    c.set_xlim(1.3, 13)
    c.set_xticks([1.5, 2, 3, 5, 7, 10])
    c.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.set_ylim(-16, 28)
    c.set_xlabel(r'central Kn at the largest core')
    c.set_ylabel('collapse-time error [%]')
    c.legend(loc='upper left', fontsize=FS.MINPT - 0.5, frameon=False, handletextpad=0.3)
    c.text(0.97, 0.95, '(c)', transform=c.transAxes, ha='right', va='top', fontsize=FS.MINPT)
    for h, mk in (('B', 'o'), ('C', 's')):
        t0 = A.halo_units(h)['t0_myr'] / 1e3
        s = load(f'solver_{h}_a1.0_nfw.npz')
        e = load(f'emulator_{h}_nfw.npz')
        tc = float(e['t_norm_emulator_C0753']) / A.S_RESCALE
        d.plot(s['t'] * t0, s['mean_rho'][:, 1], color=FS.INK, lw=1.0)
        d.plot((e['tau'] * tc * t0)[::5], e['mean_rho'][::5, 1], mk, ms=2.6, mfc='none',
               mec=FS.STATE_HUE, mew=0.7, label=f'emulator, halo {h}')
    d.plot([], [], color=FS.INK, lw=1.0, label='solver')
    d.set_yscale('log')
    d.set_xlabel('t [Gyr]')
    d.set_ylabel(r'$\bar\rho_c/\rho_s$')
    d.legend(loc='upper left', fontsize=FS.MINPT - 0.5, frameon=False)
    d.text(0.97, 0.05, '(d) NFW input', transform=d.transAxes, ha='right', fontsize=FS.MINPT)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.8)
    FS.save(fig, 'fig25_independent_haloes', [FIG])


if __name__ == '__main__':
    main()
