#!/usr/bin/env python3
"""Figure: resolution of the particle references and the closure likelihood.

(a) Collapse marker of every NSphere run, relative to the halo's reference,
    against particle number (multifidelity/ladder_analysis.json, score_D.json).
(b) chi^2 over (alpha, C) from haloes A, B, C alone and from all five haloes
    (the grids and references of scripts/closure_fit5.py; no new runs).

    python scripts/fig_closure_calibration.py [--out DIR]
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
import closure_fit5 as F  # noqa: E402

plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.4, 'legend.frameon': False, 'pdf.fonttype': 42})
HC = {'B': '#2a78d6', 'C': '#eb6834', 'D1': '#1baf7a', 'D2': '#8a5cc2'}


def main():
    lad = json.load(open(os.path.join(EXT, 'multifidelity', 'ladder_analysis.json')))
    sc = json.load(open(os.path.join(EXT, 'closure_design', 'score_D.json')))['haloes']
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    off = {'B': -0.06, 'C': -0.02, 'D1': 0.02, 'D2': 0.06}
    for h in ('B', 'C', 'D1', 'D2'):
        if h in ('B', 'C'):
            runs = [(r['N'], r['t_x10']) for r in lad[h]['runs']]
        else:
            runs = [(r['N'], r['t_x10']) for r in sc[h]['runs'] if not r['dt2']]
        ref, sig = F.REF[h]
        x = np.array([n for n, _ in runs], float) * np.exp(off[h])
        y = np.array([t for _, t in runs]) / ref
        ax[0].plot(x, y, 'o', ms=3.5, color=HC[h], label=h)
        # paired half-time-step repeats (same seed as a 1e5 run): open markers
        if h == 'C':
            import independent_halo_analysis as IH
            d3 = np.load(os.path.join(ROOT, 'results', 'independent_halo_20261107', 'nsphere_C3.npz'))
            dts = [IH.collapse_markers(d3['t'], d3['mean_rho'][:, 1])['t_x10']]
        elif h in ('D1', 'D2'):
            dts = [r['t_x10'] for r in sc[h]['runs'] if r['dt2']]
        else:   # halo B: repeat of seed 1 (B1), closure_design/score_E.json
            dts = [json.load(open(os.path.join(EXT, 'closure_design', 'score_E.json')))['B_half_time_step']['B_dt2_t_x10_t0']]
        for t in dts:
            ax[0].plot(1e5 * np.exp(off[h]) * 1.12, t / ref, 'o', ms=3.5, mfc='white', color=HC[h], zorder=5)
        ax[0].axhspan(1 - sig, 1 + sig, color=HC[h], alpha=0.06, lw=0)
    ax[0].axhline(1, color='k', lw=0.6)
    ax[0].set_xscale('log'); ax[0].set_xticks([5e4, 1e5, 2e5]); ax[0].set_xticklabels([r'$5\times10^4$', r'$10^5$', r'$2\times10^5$'])
    ax[0].minorticks_off()
    ax[0].set_xlabel('particles'); ax[0].set_ylabel(r'$t_{x10}$ / reference')
    ax[0].legend(fontsize=6, ncol=4, loc='upper center')
    ax[0].set_ylim(0.85, 1.15)
    ax[0].set_title('(a) particle references', loc='left', fontsize=7.5)

    H = list(F.REF)
    G = {h: F.grid(h) for h in H}
    fa = np.linspace(0.3, 1.5, 241)
    fc = np.linspace(np.log(min(F.CS)), np.log(max(F.CS)), 261)
    A_, C_ = np.meshgrid(fa, fc, indexing='ij')
    pts = np.stack([A_.ravel(), C_.ravel()], 1)
    R = {h: (G[h](pts) / F.REF[h][1]).reshape(A_.shape) for h in H}
    for hs, col, lab in ((('A', 'B', 'C'), '#9a9a9a', 'A, B, C'), (tuple(H), '#1baf7a', 'all five')):
        chi = sum(R[h] ** 2 for h in hs)
        m = np.nanmin(chi)
        ax[1].contourf(A_, np.exp(C_), chi - m, levels=[0, 1, 4], colors=[col, col], alpha=0.35 if len(hs) == 5 else 0.25)
        ax[1].contour(A_, np.exp(C_), chi - m, levels=[1, 4], colors=[col], linewidths=0.8)
        ax[1].plot([], [], color=col, lw=4, alpha=0.5, label=fr'{lab}: $\Delta\chi^2 = 1, 4$')
    ax[1].plot(1.0, 0.712, 's', color='#2a78d6', ms=4, label='harmonic')
    ax[1].plot(0.5, 0.9, 'D', color='#eb6834', ms=4, label=r'$\alpha=1/2$, $C=0.9$')
    ax[1].plot(0.58, 0.85, '*', color='k', ms=7, label='fit, five haloes')
    ax[1].set_xscale('log'); ax[1].set_xticks([0.3, 0.5, 0.7, 1.0, 1.5]); ax[1].set_xticklabels(['0.3', '0.5', '0.7', '1', '1.5'])
    ax[1].minorticks_off()
    ax[1].set_xlabel(r'interpolation exponent $\alpha$'); ax[1].set_ylabel(r'LMFP constant $C$')
    ax[1].legend(fontsize=5.8, loc='upper right')
    ax[1].set_title(r'(b) closure likelihood', loc='left', fontsize=7.5)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_calibration.{ext}'), dpi=200)


if __name__ == '__main__':
    main()
