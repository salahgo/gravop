"""Predicted against measured deployment penalty, on three populations.

The quadrature rule p_hat = sqrt(E_A^2 + F^2) / E_A against the measured
p = E_dep / E_A at the converged clock:
  * in sample: the 15 system-backbone points of the frozen test split
    (results/statecond/decomposition.json);
  * reserved draws: the same 15 points on 320 draws solved after the rule was
    frozen (results/e43/prospective.json, docs/PROSPECTIVE_PREREGISTRATION.md);
  * disjoint systems: six equations outside the benchmark, FNO only
    (results/track_b_20261106/tasks/score/track_b.json,
    docs/DISJOINT_PREREGISTRATION.md).
The horizontal line is the comparator constant, log10 p_c = 0.1753, fitted on
the 15 in-sample points. Reads artefacts only; writes
results/figures/fig23_penalty_prediction.{pdf,eps,png}.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import figstyle as FS

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(RES, 'figures')
FS.use('plot')

NEW_MARK = {'quench': 'v', 'pme': '<', 'cgl': '>', 'hj': 'h', 'corr': 'X', 'bimodal': '*'}


def points():
    dec = json.load(open(os.path.join(RES, 'statecond', 'decomposition.json')))
    pro = json.load(open(os.path.join(RES, 'e43', 'prospective.json')))['points']
    tb = json.load(open(os.path.join(RES, 'track_b_20261106', 'tasks', 'score', 'track_b.json')))
    log_pc = tb['log10_constant_penalty']

    def pp(v):
        return v['E_dep'] / v['E_A'], np.sqrt(v['E_A'] ** 2 + v['F'] ** 2) / v['E_A']

    ins = {k: pp(v) for k, v in dec.items()}
    res = {k: pp(v) for k, v in pro.items()}
    new = {k: pp(v['aggregate']) for k, v in tb['new'].items()}
    return ins, res, new, log_pc


def main():
    ins, res, new, log_pc = points()
    fig, ax = plt.subplots(figsize=(FS.COL1 * 1.35, FS.COL1 * 1.2))
    lo, hi = 0.75, 6.5
    xx = np.array([lo, hi])
    ax.fill_between(xx, xx / 10 ** 0.1, xx * 10 ** 0.1, color='#eeeeee', lw=0, zorder=0)
    ax.plot(xx, xx, color=FS.INK, lw=0.8, zorder=1)
    ax.axhline(10 ** log_pc, color=FS.MUTED, lw=0.8, ls='--', zorder=1)
    ax.text(hi * 0.97, 10 ** log_pc * 0.94, 'fitted constant', ha='right', va='top',
            fontsize=FS.MINPT, color=FS.MUTED)
    for k, (p, ph) in ins.items():
        ax.scatter(p, ph, marker=FS.MARK[k.rsplit('_', 1)[0]], s=22, facecolor='none',
                   edgecolor=FS.TAU_HUE, lw=0.8, zorder=3)
    for k, (p, ph) in res.items():
        ax.scatter(p, ph, marker=FS.MARK[k.rsplit('_', 1)[0]], s=22, color=FS.TAU_HUE,
                   edgecolor=FS.INK, lw=0.3, zorder=3)
    for k, (p, ph) in new.items():
        ax.scatter(p, ph, marker=NEW_MARK[k], s=40 if k == 'bimodal' else 26,
                   color=FS.STATE_HUE, edgecolor=FS.INK, lw=0.3, zorder=4)
        dx, dy, ha = {'bimodal': (-0.03, 0.0, 'right'), 'pme': (-0.025, 0.0, 'right'),
                      'corr': (-0.025, 0.0, 'right')}.get(k, (None, None, None))
        if dx is not None:
            ax.annotate(k, (p, ph), xytext=(p * 10 ** dx, ph * 10 ** dy), fontsize=FS.MINPT,
                        ha=ha, va='center', color=FS.STATE_HUE)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ticks = [0.8, 1, 1.5, 2, 3, 4, 6]
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        a.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        a.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r'measured penalty $E_{\rm dep}/E_A$')
    ax.set_ylabel(r'predicted $\sqrt{E_A^2+F^2}/E_A$')
    h = [plt.Line2D([], [], marker='o', ls='', mfc='none', mec=FS.TAU_HUE, label='benchmark, test split'),
         plt.Line2D([], [], marker='o', ls='', color=FS.TAU_HUE, mec=FS.INK, mew=0.3, label='benchmark, reserved draws'),
         plt.Line2D([], [], marker='o', ls='', color=FS.STATE_HUE, mec=FS.INK, mew=0.3, label='disjoint equations'),
         plt.Rectangle((0, 0), 1, 1, color='#eeeeee', label=r'$\pm0.1$ dex')]
    leg = ax.legend(handles=h, loc='upper left', fontsize=FS.MINPT, handletextpad=0.3)
    ax.add_artist(leg)
    # shape key: the marker identifies the system (fill and colour give the population)
    shapes = {**FS.MARK, **NEW_MARK}
    k = [plt.Line2D([], [], marker=m, ls='', mfc='none', mec=FS.INK, mew=0.6, ms=4.5, label=n)
         for n, m in shapes.items()]
    ax.legend(handles=k, loc='lower right', ncol=2, fontsize=FS.MINPT, handletextpad=0.2,
              columnspacing=0.6, labelspacing=0.25, borderpad=0.3, frameon=False,
              prop={'family': 'monospace', 'size': FS.MINPT})
    ax.grid(False)
    FS.save(fig, 'fig23_penalty_prediction', [FIG])


if __name__ == '__main__':
    main()
