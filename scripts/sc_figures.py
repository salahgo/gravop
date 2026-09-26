"""Figures for the cross-domain conditioning study.

Reads results/statecond/*.json only. Outputs results/figures/fig8_*, fig9_*.

Style follows scripts/make_figures.py. Palette: the tau-conditioned operator is
one model evaluated two ways, so both its bars carry the same hue (#eb6834) and
the unattainable oracle evaluation is hatched; the state-conditioned operator is
a different method and carries the second hue (#2a78d6). The pair validates
clean on all six checks (dataviz skill validator, light surface, 2026-08-30):
CVD deltaE 24.7 protan, normal 33.6, both above 3:1 contrast.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
FIG = os.path.join(ROOT, 'results', 'figures')
OUTDIRS = [FIG]
for _d in OUTDIRS:
    os.makedirs(_d, exist_ok=True)

import figstyle as FS

TAU_HUE, STATE_HUE = FS.TAU_HUE, FS.STATE_HUE
HYB = {'H1_joint': FS.DIRECT_HUE, 'H2_marginalised': '#eda100',
       'H3_ensemble': '#e87ba4'}
# Backbone identity in fig12, drawn from the shared categorical set in
# `figstyle`, which clears all six checks including contrast (see that module).
# Points still carry a shape as well as a hue so identity survives greyscale.
BACKBONE_HUE = {'fno': FS.STATE_HUE, 'deeponet': FS.TAU_HUE,
                'cno': FS.DIRECT_HUE, 'transformer': FS.AUX_HUE,
                # P10: the same architecture as `fno` with an added H1 loss
                # term, so it is drawn in the FNO hue but hollow, to show at a
                # glance that it is a training variant and not a fifth family.
                'fno_h1': '#8fbdf0'}
MARK = {'rd': 'o', 'ks': 's', 'film': '^', 'burgers': 'D', 'sidm': 'P'}
ORDER = ['rd', 'ks', 'film', 'burgers', 'sidm']
TITLE = {'rd': 'reaction-diffusion\n(combustion)',
         'ks': 'Keller-Segel\n(biology)',
         'film': 'thin-film rupture\n(interfaces)',
         'burgers': 'Burgers steepening\n(gas dynamics)',
         'sidm': 'gravothermal collapse\n(astrophysics)'}

FS.use('plot')


def save(fig, name):
    # EPS as well as PDF: CPC asks for EPS and this script wrote none until
    # 2026-09-12, so thirteen of the seventeen figures had no EPS at all.
    FS.save(fig, name, OUTDIRS)


def load(name, prefix=''):
    p = os.path.join(RES, f'{prefix}{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def fig_cross_domain(rows):
    """The four arms at the converged clock.

    Rebuilt 2026-09-13. The previous version plotted the deployed arm through
    the ORIGINAL time head, which was undertrained, so it displayed a penalty
    the paper no longer claims. The panel that showed error against normalised
    time is dropped rather than relabelled, because those curves exist only for
    the superseded head.
    """
    th = json.load(open(os.path.join(RES, 'timehead.json')))
    dr = json.load(open(os.path.join(RES, 'direct.json')))
    ct = json.load(open(os.path.join(RES, 'contract.json')))
    names = [n for n, _ in rows]

    fig, ax = plt.subplots(figsize=(FS.COL2, 3.1))
    xs = np.arange(len(names))
    w = 0.17
    # Arm C appears twice on purpose. Plotting it only at an exact coordinate
    # gives it information the time arms do not have; the contract bar is the
    # like-for-like comparison.
    series = [
        (-2.0 * w, 'oracle normalised time (not deployable)', TAU_HUE, 'o',
         lambda n: th[n]['armA']),
        (-1.0 * w, 'deployed normalised time', TAU_HUE, 's',
         lambda n: th[n]['armB_converged']),
        (0.0, 'direct physical time', FS.DIRECT_HUE, 'D',
         lambda n: dr[n]['nrmse']['D_time_direct']['mean']),
        (1.0 * w, 'state, exact coordinate (not deployable)', STATE_HUE, 'o',
         lambda n: th[n]['armC']),
        (2.0 * w, 'state, measurement contract', STATE_HUE, '^',
         lambda n: ct[n]['nrmse']['full']['mean']),
    ]
    # Markers on a stem, not bars. A bar on a log axis has no meaningful
    # baseline -- its length is set by wherever the axis happens to start --
    # so the eye reads a ratio that is not in the data. The stem carries the
    # position and the marker carries the value.
    lo = min(get(n) for _, _, _, _, get in series for n in names) * 0.55
    for off, lab, col, marker, get in series:
        vals = [get(n) for n in names]
        ax.vlines(xs + off, lo, vals, color=col, lw=1.0, alpha=0.55, zorder=2)
        ax.plot(xs + off, vals, linestyle='none', marker=marker, ms=7.0,
                color=col, markeredgecolor=FS.INK, markeredgewidth=0.6,
                label=lab, zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(lo, None)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel('field nRMSE  (log scale)')
    ax.legend(ncol=3, fontsize=FS.MINPT, loc='upper center',
              bbox_to_anchor=(0.5, 1.30))
    ax.grid(axis='x', visible=False)
    save(fig, 'fig8_cross_domain')

def fig_decomposition():
    """P33/P36: the three terms of the exact decomposition, per point.

    Normalised by the measured deployed error, so the three terms sum to one by
    construction and the cross term's share is read directly. The manuscript
    reported that term as a median and a count, which hides that it is negative
    almost everywhere and large in some points.
    """
    dec = json.load(open(os.path.join(RES, 'decomposition.json')))
    keys = sorted(dec, key=lambda k: (FS.SYS_ORDER.index(k.rsplit('_', 1)[0]),
                                      k))
    tot = np.array([dec[k]['E_dep'] ** 2 for k in keys])
    F2 = np.array([dec[k]['F'] ** 2 for k in keys]) / tot
    E2 = np.array([dec[k]['e_dep'] ** 2 for k in keys]) / tot
    CR = np.array([dec[k]['cross'] for k in keys]) / tot

    # Grouped, not stacked: the cross term is negative, and a stacked
    # negative segment reads as though it adds to the total.
    fig, ax = plt.subplots(figsize=(FS.COL2, 4.1))
    y = np.arange(len(keys))
    h = 0.26
    ax.barh(y - h, F2, h, color=TAU_HUE, edgecolor=FS.INK, linewidth=0.4,
            label=r'displacement $F^2$')
    ax.barh(y, E2, h, color=STATE_HUE, edgecolor=FS.INK, linewidth=0.4,
            label=r'operator error $\|e(\tilde v)\|^2$')
    ax.barh(y + h, CR, h, color=FS.BAD, edgecolor=FS.INK, linewidth=0.4,
            label=r'cross term $2\langle\Delta,e(\tilde v)\rangle$')
    ax.axvline(1.0, color=FS.INK, lw=0.9, ls='--')
    ax.text(1.02, -0.95, 'measured $E_{\\rm dep}^2$', fontsize=FS.MINPT,
            color=FS.INK, va='center')
    ax.axvline(0.0, color=FS.RULE, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace('_', '  ') for k in keys])
    ax.set_ylim(len(keys) - 0.5, -1.3)
    ax.set_xlabel(r'share of the measured $E_{\rm dep}^2$'
                  '  (the three sum to one)')
    ax.legend(ncol=3, fontsize=FS.MINPT, loc='upper center',
              bbox_to_anchor=(0.5, 1.09))
    ax.grid(axis='y', visible=False)
    save(fig, 'fig19_decomposition')


def fig_hybrid(rows):
    fig, axes = plt.subplots(1, len(rows), figsize=(FS.COL2, 2.6), sharey=False)
    if len(rows) == 1:
        axes = [axes]
    # H2 and H3 sit almost exactly on top of C -- that coincidence IS the
    # result, so C is drawn wide underneath and the two are dashed/dotted on
    # top rather than hiding it
    keys = [('B_tau_deployed', TAU_HUE, 'time-conditioned, deployed',
             '-', 1.8, 3),
            ('C_state_point', STATE_HUE, 'state, point estimate',
             '-', 3.0, 0),
            ('H1_joint', HYB['H1_joint'], 'H1 joint (time + state)',
             '-', 1.8, 3),
            ('H2_marginalised', HYB['H2_marginalised'], 'H2 marginalised',
             '--', 1.4, 0),
            ('H3_ensemble', HYB['H3_ensemble'], 'H3 ensemble',
             ':', 1.6, 0)]
    for ax, (name, r) in zip(axes, rows):
        eps = list(r['eps_grid_dex'])
        for key, col, lab, ls, lw, ms in keys:
            ys = [r['aggregate'][f'eps{e}_q0.0'][key]['mean'] for e in eps]
            ax.plot(eps, ys, color=col, label=lab, ls=ls, lw=lw,
                    marker='o' if ms else None, ms=ms,
                    # EPS has no alpha: carry the distinction with a
                    # dash pattern rather than a fade, so both formats agree.
                    **({'dashes': (1.6, 1.4)}
                       if key == 'C_state_point' else {}))
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, numticks=4))
        ax.set_title(TITLE[name], fontsize=7)
        ax.set_xticks([0.0, 0.1, 0.2])
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('field nRMSE')
    axes[-1].legend(frameon=False, fontsize=FS.MINPT, loc='center left',
                    bbox_to_anchor=(1.04, 0.5))
    fig.supxlabel('phase measurement noise (dex)', fontsize=8.5, y=-0.06)
    # the honest headline: the graceful-degradation scheme is joint
    # conditioning, not marginalisation, which tracks the point estimate
    # 2026-09-12: the caption belongs in the LaTeX caption, not baked
    # into the artwork. These suptitles were full paragraphs, and
    # fig12's asserted the spectral-bias reading that N10 retracted.
    fig.subplots_adjust(wspace=0.42)
    save(fig, 'fig9_hybrid_noise')


def fig_robust(rows, base):
    """E24: training on the deployed inputs flattens the noise response."""
    b = dict(base)
    fig, axes = plt.subplots(1, len(rows), figsize=(FS.COL2, 2.6))
    if len(rows) == 1:
        axes = [axes]
    keys = [('C_clean_trained', STATE_HUE, 'state, clean-trained', '-', 1.8),
            ('time_aug', TAU_HUE, 'time, noise-trained', '--', 1.6),
            ('phase_aug', HYB['H1_joint'], 'state, noise-trained', '-', 1.8),
            ('joint_aug', HYB['H2_marginalised'], 'joint, noise-trained',
             '-', 2.2),
            ('joint_rel', HYB['H3_ensemble'], 'joint + reliability', '--', 1.5)]
    for ax, (name, r) in zip(axes, rows):
        eps = list(r['eps_grid_dex'])
        if name in b:
            bv = b[name]['nrmse']['B_tau_deployed']['mean']
            ax.axhline(bv, color=TAU_HUE, lw=1.8,
                       label='time-conditioned, deployed')
        for key, col, lab, ls, lw in keys:
            ys = [r['aggregate'][f'eps{e}_q0.0'][key]['mean'] for e in eps]
            ax.plot(eps, ys, color=col, label=lab, ls=ls, lw=lw, marker='o',
                    ms=3)
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, numticks=4))
        ax.set_title(TITLE[name], fontsize=7)
        ax.set_xticks([0.0, 0.1, 0.2])
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('field nRMSE')
    axes[-1].legend(frameon=False, fontsize=FS.MINPT, loc='center left',
                    bbox_to_anchor=(1.04, 0.5))
    fig.supxlabel('phase measurement noise (dex)', fontsize=8.5, y=-0.06)
    # degradation across the sweep, computed rather than typed
    def spread(key):
        f = []
        for _, r in rows:
            e = list(r['eps_grid_dex'])
            v0 = r['aggregate'][f'eps{e[0]}_q0.0'][key]['mean']
            v1 = r['aggregate'][f'eps{e[-1]}_q0.0'][key]['mean']
            f.append(v1 / v0)
        return f'{min(f):.1f}-{max(f):.0f}x'
    fig.subplots_adjust(wspace=0.42)
    save(fig, 'fig10_robust_conditioning')


def fig_rollout(rows, base):
    """E25: where rollout loses is exactly where the singularity is."""
    b = dict(base)
    fig, axes = plt.subplots(1, len(rows), figsize=(FS.COL2, 2.6))
    if len(rows) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, rows):
        taus = np.array(r['tau_grid'])
        if name in b and b[name].get('curves'):
            c = b[name]['curves']
            ax.plot(taus, c['B_tau_deployed'], color=TAU_HUE, lw=1.6,
                    label='time-conditioned, deployed')
            ax.plot(taus, c['C_state'], color=STATE_HUE, lw=1.6,
                    label='state-conditioned')
        for lab, col, ls in (('R_res', HYB['H1_joint'], '--'),
                             ('R_push', HYB['H1_joint'], '-')):
            if lab in r['arms']:
                ax.plot(taus[1:], r['arms'][lab]['curve'][1:], color=col,
                        ls=ls, lw=1.6,
                        label=f"rollout ({'pushforward' if lab == 'R_push' else 'residual'})")
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, numticks=4))
        ax.set_title(TITLE[name], fontsize=7)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('field nRMSE')
    axes[-1].legend(frameon=False, fontsize=FS.MINPT, loc='center left',
                    bbox_to_anchor=(1.04, 0.5))
    fig.supxlabel(r'$\tau = t/t_{\rm sing}$', fontsize=8.5, y=-0.06)
    fig.subplots_adjust(wspace=0.42)
    save(fig, 'fig11_rollout')


def fig_law(law):
    """E27/P1: the deployment penalty is set by how sharply the operator
    resolves phase, so a *better* operator is penalised *more*."""
    pts = law['points']
    f, ci = law['fit_all'], law['exponent_ci']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(FS.COL2, 3.2))

    x = np.array([p['sensitivity'] for p in pts])
    y = np.array([p['penalty'] for p in pts])
    gx = np.logspace(np.log10(x.min() * 0.8), np.log10(x.max() * 1.25), 50)
    lg = np.log10(gx)
    ax.plot(gx, 10 ** (f['exponent'] * lg + f['intercept']), color='#555555',
            lw=1.2, zorder=1,
            label=(f"fit: $p \\propto s^{{{f['exponent']:.2f}}}$, "
                   f"$r={f['r_log']:.3f}$"))
    ax.fill_between(gx, 10 ** (ci['lo'] * lg + f['intercept']),
                    10 ** (ci['hi'] * lg + f['intercept']),
                    color='#eaeae9', lw=0, zorder=0,
                    label='95% CI on the exponent')
    for p in pts:
        ax.plot(p['sensitivity'], p['penalty'], MARK[p['system']],
                color=BACKBONE_HUE[p['backbone']], ms=7,
                mec='#fcfcfb', mew=0.9, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('phase sensitivity $s$  (error at 10% imposed\n'
                  '$t_{\\rm sing}$ error, / oracle error)')
    ax.set_ylabel('deployment penalty $p$  (B/A)')
    ax.legend(frameon=False, fontsize=FS.MINPT, loc='upper left')
    for a in (ax, ax2):
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # the P1 prediction, read directly: within each system, connect the
    # backbones in order of accuracy
    for name in ORDER:
        sub = sorted([p for p in pts if p['system'] == name],
                     key=lambda q: -q['oracle'])
        if not sub:
            continue
        ax2.plot([p['oracle'] for p in sub], [p['penalty'] for p in sub],
                 '-', color='#bbbbbb', lw=1.0, zorder=1)
        for p in sub:
            ax2.plot(p['oracle'], p['penalty'], MARK[name],
                     color=BACKBONE_HUE[p['backbone']], ms=7, mec='#fcfcfb',
                     mew=0.9, zorder=3)
        ax2.annotate(f' {name}', (sub[-1]['oracle'], sub[-1]['penalty']),
                     fontsize=FS.MINPT, color='#555555', va='center')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    # headroom on the left of the inverted axis, where the per-system labels
    # sit next to the most accurate backbone
    ox = [p['oracle'] for p in pts]
    ax2.set_xlim(max(ox) * 1.35, min(ox) * 0.55)
    ax2.set_xlabel('oracle nRMSE  (more accurate $\\rightarrow$)')
    ax2.set_ylabel('deployment penalty $p$  (B/A)')

    handles = [plt.Line2D([], [], marker='o', ls='', color=c, ms=6,
                          mec='#fcfcfb', mew=0.8, label=b)
               for b, c in BACKBONE_HUE.items()
               if any(p['backbone'] == b for p in pts)]
    handles += [plt.Line2D([], [], marker=MARK[n], ls='', color='#888888',
                           ms=6, label=n) for n in ORDER
                if any(p['system'] == n for p in pts)]
    ax2.legend(handles=handles, frameon=False, fontsize=FS.MINPT, ncol=1,
               loc='center left', bbox_to_anchor=(1.03, 0.5))
    fig.subplots_adjust(wspace=0.38)
    save(fig, 'fig12_sensitivity_law')


def main():
    base = [(n, load(n)) for n in ORDER]
    base = [(n, r) for n, r in base if r]
    if base:
        fig_cross_domain(base)
    else:
        print('no base results yet')
    if os.path.exists(os.path.join(RES, 'decomposition.json')):
        fig_decomposition()
    hyb = [(n, load(n, 'hybrid_')) for n in ORDER]
    hyb = [(n, r) for n, r in hyb if r]
    if hyb:
        fig_hybrid(hyb)
    else:
        print('no hybrid results yet')
    rob = [(n, load(n, 'robust_')) for n in ORDER]
    rob = [(n, r) for n, r in rob if r]
    if rob:
        fig_robust(rob, base)
    else:
        print('no robust results yet')
    rol = [(n, load(n, 'rollout_')) for n in ORDER]
    rol = [(n, r) for n, r in rol if r]
    if rol:
        fig_rollout(rol, base)
    else:
        print('no rollout results yet')
    p = os.path.join(RES, 'law.json')
    if os.path.exists(p):
        fig_law(json.load(open(p)))
    else:
        print('no law.json yet — run scripts/sc_law.py')


if __name__ == '__main__':
    main()
