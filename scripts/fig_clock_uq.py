#!/usr/bin/env python3
"""Figure: clock-error propagation for the baryon-aware emulator.

Recomputes, without writing to the experiment records, the per-query arrays of
experiments/research_extensions/scripts/clock_uq.py (same members, splits and
calibration) and draws
  (a) the emulator's collapse clock against the solver's on test haloes;
  (b) ROC curves for detecting queries with relative density error > 10%;
  (c) median absolute error against the attached uncertainty (calibration).
Checks the AUROC values against clock_uq/clock_uq.json.

    python scripts/fig_clock_uq.py [--out DIR]      (GPU, about 1 min)
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
import clock_uq as U  # noqa: E402

plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.4, 'legend.frameon': False, 'pdf.fonttype': 42})
COL = {'s_base': '#2a78d6', 's_own': '#eda100', 's_prop': '#1baf7a'}
LAB = {'s_base': 'ensemble spread', 's_own': 'members on own clocks', 's_prop': 'clock-error propagation'}


def roc(score, bad):
    o = np.argsort(-score)
    y = bad[o]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(~y) / (~y).sum()])
    return fpr, tpr


def main():
    import train_baryon_emulator as T
    from gravop import dataset as ds
    d = T.load()
    tr, va, te, ho = T.splits(d)
    q = ds.tau_grid()
    members = [U.load_member(os.path.join(ROOT, f'results/baryons_20261107/runs/seed{s}/model.pt')) for s in range(3)]
    v, s = U.run_split(d, va, members, q), U.run_split(d, te, members, q)
    sigma_c = float(np.sqrt(np.mean(np.log(v['c_bar'] / v['c_true']) ** 2)))
    s['s_prop'] = np.sqrt(s['s_base'] ** 2 + (s['dmu'] * sigma_c) ** 2)
    sup = np.broadcast_to(s['support'][:, :, None], s['mu'].shape)
    e_rel = np.abs(10 ** (s['mu'] - s['truth']) - 1)[sup]
    e_dex = np.abs(s['mu'] - s['truth'])[sup]
    bad = e_rel > 0.10
    ref = json.load(open(os.path.join(EXT, 'clock_uq', 'clock_uq.json')))['splits']['test']['query_level']['auroc']
    for k in COL:
        a = U.auroc(s[k][sup], bad)
        print(k, round(a, 4), 'record', round(ref[k], 4))
        assert abs(a - ref[k]) < 2e-3

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax[0].loglog(s['c_true'], s['c_bar'], 'o', ms=2.5, color='#555555')
    lo, hi = s['c_true'].min() * 0.8, s['c_true'].max() * 1.25
    ax[0].plot([lo, hi], [lo, hi], 'k-', lw=0.6)
    ax[0].set_xlabel(r'solver $t_{\rm end}/t_0$'); ax[0].set_ylabel(r'emulator $\hat t_{\rm end}/t_0$')
    ax[0].set_title('(a) terminal time, test haloes', loc='left', fontsize=7.5)
    for k in COL:
        f, t = roc(s[k][sup], bad)
        ax[1].plot(f, t, color=COL[k], label=f"{LAB[k]} ({U.auroc(s[k][sup], bad):.2f})")
    ax[1].plot([0, 1], [0, 1], color='#bbbbbb', lw=0.6)
    ax[1].set_xlabel('false-positive rate'); ax[1].set_ylabel('true-positive rate')
    ax[1].legend(fontsize=5.8, loc='lower right')
    ax[1].set_title('(b) detection of bad queries', loc='left', fontsize=7.5)
    for k in ('s_base', 's_prop'):
        x = s[k][sup]
        edges = np.quantile(x, np.linspace(0, 1, 16))
        idx = np.clip(np.digitize(x, edges) - 1, 0, 14)
        xm = np.array([np.median(x[idx == i]) for i in range(15)])
        ym = np.array([np.median(e_dex[idx == i]) for i in range(15)])
        ax[2].loglog(xm, ym, 'o-', ms=3, color=COL[k], label=LAB[k])
    g = np.array([1e-3, 1])
    ax[2].plot(g, g, 'k-', lw=0.6)
    ax[2].set_xlim(1e-3, 0.5); ax[2].set_ylim(1e-3, 0.5)
    ax[2].set_xlabel('attached uncertainty (dex)'); ax[2].set_ylabel('median |error| (dex)')
    ax[2].legend(fontsize=5.8, loc='upper left')
    ax[2].set_title('(c) error against uncertainty', loc='left', fontsize=7.5)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_clock_uq.{ext}'), dpi=200)
    print('sigma_c', sigma_c)


if __name__ == '__main__':
    main()
