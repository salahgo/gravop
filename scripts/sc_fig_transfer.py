"""Figure for P11: does operator accuracy reach deployment?

Left panel: deployed error against oracle error, per system, across backbones,
with the eta = 1 reference line that fully-transferring accuracy would follow.
A system whose points lie flat has stopped converting operator quality into
deployed quality; a system whose points slope upwards has inverted it.

Right panel: the same thing without any fit -- the spread in oracle error across
backbones against the spread in deployed error over the same backbones. Points
on the diagonal transfer accuracy fully; points pinned to the bottom do not.

Reads `results/statecond/law.json` and `results/statecond/transfer.json`.
Usage:  python scripts/sc_fig_transfer.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
FIG = os.path.join(ROOT, 'results', 'figures')
OUTDIRS = [FIG]

SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']
LABEL = {'rd': 'reaction-diffusion', 'ks': 'Keller-Segel',
         'film': 'thin-film', 'burgers': 'Burgers (control)',
         'sidm': 'gravothermal'}
COLOR = {'rd': '#1b6ca8', 'ks': '#c1121f', 'film': '#2a9d8f',
         'burgers': '#8d99ae', 'sidm': '#6a4c93'}
MARK = {'deeponet': 'o', 'fno': 's', 'transformer': '^', 'cno': 'D',
        'fno_h1': 'v'}
# hand-placed so the five labels in panel (b) do not collide; they cluster
# Panel (b) direct labels. 'sidm' sits lowest in the panel, so a downward
# offset put its label across the axis; it is labelled above instead, and
# 'film' moves down to keep the two clear of each other.
OFFSET = {'rd': (0, 11, 'center'), 'ks': (10, 4, 'left'),
          'film': (0, 25, 'center'), 'burgers': (0, -16, 'center'),
          'sidm': (0, -15, 'center')}

import figstyle as FS

FS.use('plot')


def main():
    law = json.load(open(os.path.join(RES, 'law.json')))['points']
    tr = json.load(open(os.path.join(RES, 'transfer.json')))['per_system']

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    for s in SYSTEMS:
        sub = sorted([p for p in law if p['system'] == s],
                     key=lambda p: p['oracle'])
        if not sub:
            continue
        o = np.array([p['oracle'] for p in sub])
        d = np.array([p['oracle'] * p['penalty'] for p in sub])
        c = COLOR[s]
        # EPS has no alpha channel; a thinner stroke recedes instead.
        axA.plot(o, d, '-', color=c, lw=1.1, zorder=2)
        for p, oo, dd in zip(sub, o, d):
            axA.plot(oo, dd, MARK.get(p['backbone'], 'o'), color=c, ms=5,
                     mec='white', mew=0.6, zorder=3)
        # The inline label used to repeat the eta already carried by the
        # legend entry, and for the control it landed on the eta = 1 reference
        # line. The legend states name and eta together, so the duplicate goes.
        eta = tr[s]['eta']
        axA.plot([], [], '-o', color=c, ms=4,
                 label=f'{LABEL[s]}  ($\\eta$={eta:+.2f})')

    # the eta = 1 reference: accuracy transferring fully to deployment
    xs = np.array([2.5e-3, 1.6e-1])
    axA.plot(xs, xs * 2.6, ':', color='0.35', lw=1.2, zorder=1)
    axA.annotate('$\\eta=1$', xy=(3.4e-3, 8.0e-3), color='0.35',
                 fontsize=8, style='italic')
    axA.legend(loc='lower left', frameon=False, handlelength=1.4,
               borderaxespad=0.2)

    axA.set_xscale('log')
    axA.set_yscale('log')
    axA.set_xlabel('oracle error (nRMSE) — better operator $\\rightarrow$')
    axA.set_ylabel('deployed error (nRMSE)')
    axA.invert_xaxis()
    axA.set_title('(a) deployed against oracle error', loc='left')

    # ---- right: the fit-free version
    for s in SYSTEMS:
        v = tr[s]
        axB.plot(v['oracle_range_factor'], v['deployed_range_factor'], 'o',
                 color=COLOR[s], ms=8, mec='white', mew=0.8, zorder=3)
        dx, dy, ha = OFFSET[s]
        axB.annotate(LABEL[s],
                     xy=(v['oracle_range_factor'], v['deployed_range_factor']),
                     xytext=(dx, dy), textcoords='offset points',
                     ha=ha, fontsize=7.5, color=COLOR[s])
    lim = np.array([1.0, 10.0])
    axB.plot(lim, lim, ':', color='0.35', lw=1.2, zorder=1)
    axB.annotate('full transfer', xy=(4.3, 4.9), rotation=38, color='0.35',
                 fontsize=7.5)
    axB.axhline(1.0, color='0.6', lw=0.8, ls='--', zorder=1)
    axB.annotate('no transfer', xy=(8.4, 1.03), color='0.45', fontsize=7.5,
                 ha='right', va='bottom')
    axB.set_xlim(2.0, 9.5)
    # room below the 'no transfer' line for the lowest system's label
    axB.set_ylim(0.55, 6.2)
    axB.set_xlabel('spread in oracle error across backbones')
    axB.set_ylabel('spread in deployed error')
    axB.set_title('(b) the same, with no fit', loc='left')

    fig.tight_layout()
    FS.save(fig, 'fig13_accuracy_transfer', OUTDIRS)


if __name__ == '__main__':
    main()
