#!/usr/bin/env python3
"""Figure: conductivity-closure test against particle references.

(a) Closure sensitivity: collapse time under alternative closures relative to the
    harmonic closure, against the central Knudsen number at the largest core
    (fluid design scan, experiments/research_extensions/closure_design/design_scan.json).
(b) Collapse-time error of three closures on the five reference haloes
    (experiments/research_extensions/learned_closures/closure_fit5.json).

Run from the repository root with the project virtual environment:
    python scripts/fig_closure_test.py [--out DIR]
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXT = os.path.join(ROOT, 'experiments', 'research_extensions')
OUT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(ROOT, 'results', 'figures')
sys.path[:0] = [os.path.join(EXT, 'scripts'), os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]

plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.6, 'legend.frameon': False, 'pdf.fonttype': 42})
COL = {'harmonic': '#2a78d6', 'literature': '#eb6834', 'fit': '#1baf7a', 'fit5': '#8a5cc2'}
LAB = {'harmonic': r'harmonic, $\alpha=1$, $C=0.712$',
       'literature': r'$\alpha=1/2$, $C=0.9$ (vdB&T)',
       'fit5': r'five-halo fit, $\alpha=0.58$, $C=0.85$',
       'fit': r'seven-halo fit, $\alpha=0.535$, $C=0.886$'}
KN = {'A': 2.26, 'B': 6.7, 'C': 1.8}  # harmonic-run Kn_c at the largest core (transfer.json, records)


def kn_values():
    pre = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))['haloes']
    tr = json.load(open(os.path.join(ROOT, 'results', 'closure_20261108', 'transfer.json')))['haloes']
    kn = {h: tr[h]['harmonic']['Kn_c_at_largest_core'] for h in ('A', 'B', 'C')}
    kn.update({h: pre[h]['predictions']['harmonic']['Kn_c'] for h in ('D1', 'D2')})
    pe = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_E.json')))['haloes']
    kn.update({h: pe[h]['predictions']['harmonic']['Kn_c'] for h in ('E1', 'E2')})
    return kn


def panel_a(ax, kn):
    d = json.load(open(os.path.join(EXT, 'closure_design', 'design_scan.json')))['results']
    harm = {(r['sigma_hat'], r['falloff']): r for r in d if r['closure'] == 'harmonic' and r['t_x10']}
    for name, col, lab in (('literature', COL['literature'], r'$\alpha=1/2$, $C=0.9$'),
                           ('valley_a0.56', COL['fit'], r'$\alpha=0.56$, $C=0.85$'),
                           ('valley_a1.5', '#8a8a8a', r'$\alpha=1.5$, $C=0.70$')):
        x, y = [], []
        for r in d:
            k = (r['sigma_hat'], r['falloff'])
            if r['closure'] == name and r['t_x10'] and k in harm:
                x.append(harm[k]['Kn_c']); y.append(r['t_x10'] / harm[k]['t_x10'])
        o = np.argsort(x)
        ax.plot(np.array(x)[o], np.array(y)[o], 'o', ms=2.5, color=col, label=lab)
    ax.axhline(1, color='k', lw=0.6)
    for h, v in kn.items():
        ax.axvline(v, color='#bbbbbb', lw=0.6, zorder=0)
        ax.text(v * {'D2': 0.9, 'E1': 1.12}.get(h, 1.0), 1.62, h, ha='center', fontsize=6)
    ax.set_xscale('log'); ax.set_ylim(0.75, 1.7)
    ax.set_xlabel(r'central Knudsen number $\mathrm{Kn}_c$')
    ax.set_ylabel(r'$t_{\rm coll}$ / harmonic')
    ax.legend(fontsize=5.8, loc='center right')
    ax.set_title('(a) fluid model: closure sensitivity', loc='left', fontsize=7.5)


def panel_b(ax, kn):
    f = json.load(open(os.path.join(EXT, 'learned_closures', 'closure_fit7.json')))
    H = ['A', 'B', 'C', 'D1', 'D2', 'E1', 'E2']
    x = np.array([kn[h] for h in H])
    sig = np.array([100 * f['refs'][h][1] for h in H])
    # E1 and E2 were predicted before they were run: for every closure that
    # was frozen, plot the pre-registered direct solver prediction (score_E),
    # not the grid interpolation, so the points match the E table.
    sE = json.load(open(os.path.join(EXT, 'closure_design', 'score_E.json')))['haloes']
    frozen = {'harmonic': 'harmonic', 'literature': 'literature', 'fit5': 'frozen_fit'}
    for j, key in enumerate(('harmonic', 'literature', 'fit5', 'fit')):
        res = dict(f['all7']['resid_pct'] if key == 'fit' else f[key]['resid_pct'])
        if key in frozen:
            res.update({h: sE[h]['closures'][frozen[key]]['pred_err_pct'] for h in ('E1', 'E2')})
        y = np.array([res[h] for h in H])
        ax.errorbar(x * (1 + 0.04 * (j - 1.5)), y, yerr=sig, fmt='osD^'[j], ms=3.5, color=COL[key], capsize=1.5,
                    elinewidth=0.8, label=LAB[key])
    ax.axhline(0, color='k', lw=0.6)
    for h in H:
        ax.text(kn[h] * (0.94 if h == 'D2' else 1.16 if h == 'E1' else 1), 19, h, ha='center', fontsize=6)
    ax.set_xscale('log'); ax.set_ylim(-32, 22)
    ax.set_xticks([0.5, 1, 2, 4, 7]); ax.set_xticklabels(['0.5', '1', '2', '4', '7'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r'central Knudsen number $\mathrm{Kn}_c$')
    ax.set_ylabel('collapse-time error (%)')
    ax.legend(fontsize=5.5, loc='lower right')
    ax.set_title('(b) fluid model vs references', loc='left', fontsize=7.5)


def main():
    kn = kn_values()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    panel_a(axs[0], kn); panel_b(axs[1], kn)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_closure.{ext}'), dpi=200)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
