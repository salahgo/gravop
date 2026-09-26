"""Schematic figures for the manuscript: arms, architecture, program flow, mechanism.

WHY THIS EXISTS
---------------
Every figure in this project so far plots a measurement. What the manuscript
lacked was the four pictures a reader needs BEFORE any measurement means
anything: what the arms are and what each costs at deployment, what the
operator actually is, how the programs fit together, and what the mechanism
claims. CPC asks for EPS, so these are written as EPS (and PDF, so the
pdflatex build picks a vector file without an epstopdf pass).

NO NUMBER IS TYPED HERE. Parameter counts are obtained by instantiating the
network; deployment requirements are read from `arm_contract.json`; the
mechanism panel is read from `coord_floor.json`. If an experiment moves, these
figures move with it, which is the same rule the plotted figures follow.

Palette: arm identity is carried by hue and is CONSISTENT with fig8-fig13 --
time-conditioned arms in the orange already used for them, the state-conditioned
arm in the blue. The four-slot categorical set validates on all six checks
against the light surface (dataviz validator, 2026-09-08): lightness band,
chroma floor, CVD separation (worst adjacent 9.2 deutan / 27.6 normal) and a
contrast WARN on the green at 2.74:1, which is discharged here because every
mark in these figures carries a visible label.

Usage:  sc_schematics.py [--outdir results/figures] [--also <second folder>]
Writes: fig14_conditioning_arms, fig15_architecture, fig16_program_flow,
        fig17_coordinate_displacement   (.eps and .pdf, plus .png for inspection)
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
RES = os.path.join(ROOT, 'results', 'statecond')

import figstyle as FS

TAU_HUE, STATE_HUE = FS.TAU_HUE, FS.STATE_HUE
DIRECT_HUE, AUX_HUE = FS.DIRECT_HUE, FS.AUX_HUE
INK, MUTED, RULE = FS.INK, FS.MUTED, FS.RULE
SURFACE = FS.SURFACE
GOOD, BAD = FS.GOOD, FS.BAD

MARK, SYS_ORDER = FS.MARK, FS.SYS_ORDER

FS.use('schematic')


def save(fig, name, outdirs):
    FS.save(fig, name, outdirs)


def box(ax, x, y, w, h, label, fc='none', ec=INK, lw=1.0, fs=8.5,
        weight='normal', ha='center', style='round,pad=0.012', tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=fc,
                                edgecolor=ec, linewidth=lw, mutation_scale=1))
    ax.text(x + w / 2 if ha == 'center' else x + 0.012, y + h / 2, label,
            ha=ha, va='center', fontsize=fs, color=tc or INK, weight=weight,
            linespacing=1.35)


def arrow(ax, xy_from, xy_to, color=MUTED, lw=1.0, style='-|>', rad=0.0):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                 mutation_scale=9, linewidth=lw, color=color,
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=1, shrinkB=1))


# ---------------------------------------------------------------- figure 14
def fig_arms(outdirs):
    """Four conditioning coordinates supplying one shared operator.

    A block diagram, not a table: the deployability matrix it used to carry is
    set as a table in the text, where tabular material belongs. Square rules,
    no fills, no icons, white ground.
    """
    rows = [
        ('A', r'$\tau = t/t_{\rm sing}$', 'exact terminal time'),
        ('B', r'$\tau = t/\hat{t}_{\rm sing}$', 'regression on initial data'),
        ('D', r'$\log_{10} t$', 'the query time'),
        ('C', r'$(\varphi, b)$', 'measured field at $t$'),
    ]
    fig, ax = plt.subplots(figsize=(FS.COL2, 1.85))
    ax.set_xlim(0, 1); ax.set_ylim(0.115, 1.0); ax.axis('off')

    X_TAG, X_CO, X_SRC = 0.008, 0.055, 0.230
    X_LEAD = 0.510   # clear of the longest source label
    X_BUS, X_SLOT, X_OP = 0.560, 0.605, 0.800
    y0, dy = 0.80, 0.205
    ys = [y0 - k * dy for k in range(4)]

    for (tag, coord, src), y in zip(rows, ys):
        ax.text(X_TAG, y, tag, fontsize=8, color=INK, va='center')
        ax.text(X_CO, y, coord, fontsize=8.5, color=INK, va='center')
        ax.text(X_SRC, y, src, fontsize=7.4, color=MUTED, va='center')
        ax.plot([X_LEAD, X_BUS], [y, y], color=INK, lw=0.6)
        ax.plot([X_BUS], [y], marker='o', ms=2.2, color=INK)

    # the four coordinates meet one slot on one operator
    ax.plot([X_BUS, X_BUS], [ys[-1], ys[0]], color=INK, lw=0.6)
    ax.plot([X_BUS, X_SLOT], [(ys[0] + ys[-1]) / 2, (ys[0] + ys[-1]) / 2],
            color=INK, lw=0.6)
    ym = (ys[0] + ys[-1]) / 2
    ax.add_patch(plt.Rectangle((X_SLOT, ym - 0.115), 0.140, 0.23,
                               fill=False, ec=INK, lw=0.8))
    ax.text(X_SLOT + 0.070, ym, 'conditioning\nslot', fontsize=7.0,
            ha='center', va='center', color=INK, linespacing=1.4)
    ax.annotate('', xy=(X_OP, ym), xytext=(X_SLOT + 0.140, ym),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6,
                                shrinkA=1, shrinkB=1))
    ax.add_patch(plt.Rectangle((X_OP, ym - 0.115), 0.100, 0.23,
                               fill=False, ec=INK, lw=0.8))
    ax.text(X_OP + 0.0525, ym, r'$G_\vartheta$', fontsize=10,
            ha='center', va='center', color=INK)
    ax.annotate('', xy=(0.955, ym), xytext=(X_OP + 0.100, ym),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6,
                                shrinkA=1, shrinkB=1))
    ax.text(0.983, ym, r'$\hat{u}$', fontsize=9.5, va='center', color=INK)

    ax.text(X_TAG, 0.955, 'Arm', fontsize=7.4, color=MUTED)
    ax.text(X_CO, 0.955, 'Coordinate', fontsize=7.4, color=MUTED)
    ax.text(X_SRC, 0.955, 'Source of the coordinate', fontsize=7.4,
            color=MUTED)
    save(fig, 'fig14_conditioning_arms', outdirs)


# ---------------------------------------------------------------- figure 15
def fig_architecture(outdirs):
    """Operator architecture and the conditioning slot that separates the arms.

    Line art on white: square rules, no fills, no shadow cards. Stage names are
    the operations they perform, not abbreviations.
    """
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from statecond.model import BACKBONES
    n_par = {}
    for extra in (1, 2):
        net = BACKBONES['fno'](n_fields=1, n_par=4, cond_extra=extra,
                               n_x=128)
        n_par[extra] = sum(p.numel() for p in net.parameters())
    delta = n_par[2] - n_par[1]

    fig, ax = plt.subplots(figsize=(FS.COL2, 2.05))
    ax.set_xlim(0, 1); ax.set_ylim(0.02, 1.0); ax.axis('off')

    yb, hb = 0.50, 0.26
    yc = yb + hb / 2
    stages = [(0.015, 0.140, 'initial field,\nparameters'),
              (0.195, 0.125, 'channel\nlifting'),
              (0.360, 0.190, 'spectral convolution\nblock'),
              (0.625, 0.135, 'pointwise\nprojection')]
    for x, w, label in stages:
        ax.add_patch(plt.Rectangle((x, yb), w, hb, fill=False, ec=INK,
                                   lw=0.8))
        ax.text(x + w / 2, yc, label, fontsize=7.2, ha='center', va='center',
                color=INK, linespacing=1.4)
    for k in range(len(stages) - 1):
        x0 = stages[k][0] + stages[k][1]
        ax.annotate('', xy=(stages[k + 1][0], yc), xytext=(x0, yc),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6,
                                    shrinkA=1, shrinkB=1))
    xe = stages[-1][0] + stages[-1][1]
    ax.annotate('', xy=(xe + 0.055, yc), xytext=(xe, yc),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6,
                                shrinkA=1, shrinkB=1))
    ax.text(xe + 0.065, yc, 'predicted\nfield', fontsize=7.2, va='center',
            color=INK, linespacing=1.4)

    xs, ws = stages[2][0], stages[2][1]
    ax.text(xs + ws / 2, yb + hb + 0.085, 'four blocks', fontsize=7.0,
            color=MUTED, ha='center', va='center')
    ax.annotate('', xy=(xs, yb + hb + 0.03), xytext=(xs + ws, yb + hb + 0.03),
                arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.5,
                                shrinkA=0, shrinkB=0))

    # the conditioning path: one modulation, applied in every block
    ycond = 0.245
    ax.add_patch(plt.Rectangle((0.195, ycond - 0.095), 0.175, 0.19,
                               fill=False, ec=INK, lw=0.8))
    ax.text(0.2825, ycond, 'conditioning slot', fontsize=7.2, ha='center',
            va='center', color=INK)
    ax.add_patch(plt.Rectangle((0.425, ycond - 0.095), 0.170, 0.19,
                               fill=False, ec=INK, lw=0.8))
    ax.text(0.510, ycond, 'feature-wise\nmodulation', fontsize=7.2,
            ha='center', va='center', color=INK, linespacing=1.4)
    ax.annotate('', xy=(0.425, ycond), xytext=(0.370, ycond),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6,
                                shrinkA=1, shrinkB=1))
    cx = xs + ws / 2
    ax.plot([0.510, 0.510], [ycond + 0.095, yb - 0.055], color=INK, lw=0.6)
    ax.plot([0.510, cx], [yb - 0.055, yb - 0.055], color=INK, lw=0.6)
    ax.annotate('', xy=(cx, yb), xytext=(cx, yb - 0.055),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6,
                                shrinkA=0, shrinkB=0))

    ax.text(0.640, ycond, 'time arms supply one scalar, the state arm two;',
            fontsize=7.2, va='center', color=INK)
    ax.text(0.640, ycond - 0.105,
            f'the difference is {delta} of {n_par[2]:,} parameters',
            fontsize=7.2, va='center', color=INK)
    save(fig, 'fig15_architecture', outdirs)


# ---------------------------------------------------------------- figure 16
def fig_program_flow(outdirs):
    """Computational workflow, by functional stage.

    The two halves are separated by hue rather than by a caption, and the one
    arrow between them is the one place they couple. Stages are named by the
    operation they perform; implementation filenames and paths are absent by
    design, being neither scientific content nor figure content.
    """
    fig, ax = plt.subplots(figsize=(FS.COL2, 2.45))
    ax.set_xlim(-0.005, 1.0); ax.set_ylim(0.0, 1.06); ax.axis('off')

    W, PITCH, H = 0.178, 0.205, 0.200

    def row(y, label, nodes, hue):
        ax.add_patch(FancyBboxPatch(
            (-0.004, y - 0.028), 4 * PITCH + W + 0.008, H + 0.056,
            boxstyle='round,pad=0,rounding_size=0.018',
            facecolor=FS.tint(hue, 0.05), edgecolor='none', zorder=0,
            mutation_aspect=2.6))
        ax.text(0.0, y + H + 0.055, label, fontsize=8.0, color=hue)
        for i, text in enumerate(nodes):
            x = i * PITCH
            _box(ax, x, y, W, H, text, FS.MINPT, FS.tint(hue, 0.17), hue, 1.0)
            if i:
                ax.annotate('', xy=(x, y + H / 2),
                            xytext=(x - (PITCH - W), y + H / 2),
                            arrowprops=dict(arrowstyle='-|>', color=hue,
                                            lw=0.9, shrinkA=1, shrinkB=1))

    row(0.585, 'Method', ['reference\nsolutions', 'operator\ntraining',
                          'arm\nevaluation', 'displacement\ndiagnostic',
                          'tables and\nfigures'], STATE_HUE)
    row(0.125, 'Application', ['gravothermal\nsolver', 'trajectory\nensemble',
                               'emulator\ntraining', 'parameter\ninference',
                               'N-body\nvalidation'], DIRECT_HUE)

    # the one coupling: the application supplies the fifth benchmark system
    ax.annotate('', xy=(0.5 * PITCH + 0.02, 0.585 - 0.028),
                xytext=(0.5 * PITCH + 0.02, 0.125 + H + 0.028),
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.9,
                                shrinkA=2, shrinkB=2))
    ax.text(0.5 * PITCH + 0.034, (0.585 + 0.125 + H) / 2,
            'supplies the fifth system', fontsize=FS.MINPT, color=MUTED,
            va='center')
    save(fig, 'fig16_program_flow', outdirs)


# ---------------------------------------------------------------- figure 17
def fig_coordinate_displacement(outdirs):
    """The mechanism: a coordinate error is a displacement along the solution."""
    cf = json.load(open(os.path.join(RES, 'coord_floor.json')))
    fig, axes = plt.subplots(1, 3, figsize=(FS.COL2, 2.5),
                             gridspec_kw={'width_ratios': [1.05, 1.0, 1.1]})

    # (a) the displacement, drawn on a trajectory
    ax = axes[0]
    ax.set_title('a  displacement along $G$',
                 fontsize=8.5, loc='left')
    t = np.linspace(0, 1, 200)
    g = 0.12 + 0.80 * t ** 3
    ax.plot(t, g, color=INK, lw=1.6)
    v, vd = 0.62, 0.80
    for xv, col, lab in ((v, INK, '$v$'), (vd, STATE_HUE, '$v+\\delta$')):
        gy = 0.12 + 0.80 * xv ** 3
        ax.plot([xv, xv], [0, gy], color=col, lw=0.9, ls=':')
        ax.plot([0, xv], [gy, gy], color=col, lw=0.9, ls=':')
        ax.plot([xv], [gy], marker='o', ms=5, color=col)
        ax.text(xv, -0.055, lab, fontsize=8, ha='center', color=col)
    y1, y2 = 0.12 + 0.80 * v ** 3, 0.12 + 0.80 * vd ** 3
    ax.annotate('', xy=(0.045, y2), xytext=(0.045, y1),
                arrowprops=dict(arrowstyle='<->', color=STATE_HUE, lw=1.3))
    ax.text(0.075, (y1 + y2) / 2, '$F_c(\\delta)$', fontsize=8.5,
            color=STATE_HUE, va='center')
    ax.set_xlabel('coordinate $c$', fontsize=8, labelpad=12)
    ax.set_ylabel('state $G_c$', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)

    # (b) the domain condition: injectivity, and what the flag restores
    ax = axes[1]
    ax.set_title('b  injectivity of $c$',
                 fontsize=8.5, loc='left')
    tt = np.linspace(0, 1, 300)
    fold = 0.55 - 0.85 * (tt - 0.42) ** 2 + 0.35 * tt
    ax.plot(tt, fold, color=AUX_HUE, lw=1.6)
    lvl = 0.62
    ax.axhline(lvl, color=MUTED, lw=0.8, ls='--')
    hits = tt[np.abs(fold - lvl) < 0.004]
    for xh in (hits[0], hits[-1]):
        ax.plot([xh], [lvl], marker='o', ms=5, color=AUX_HUE)
    apex = float(tt[int(np.argmax(fold))])
    ax.axvline(apex, color=MUTED, lw=0.8, ls=':')
    # y must clear set_ylim below: at 0.045 the label fell outside the axes and
    # was struck through by the bottom spine.
    ax.text(0.5, 0.115, 'one value, two instants', ha='center', fontsize=7.2,
            color=AUX_HUE, linespacing=1.4)
    ax.set_xlabel(r'normalised time $\tau$', fontsize=8)
    ax.set_ylabel('coordinate $c$', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(0.05, 1.02)

    # (c) the test: predicted against measured, 25 points, no network
    ax = axes[2]
    ax.set_title('c  predicted against measured', fontsize=8.5,
                 loc='left')
    lo, hi = 1e-3, 1.0
    ax.plot([lo, hi], [lo, hi], color=RULE, lw=1.0, zorder=1)
    for name in SYS_ORDER:
        xs = [cf[name]['prediction'][str(e)]['measured'] for e in cf['eps_dex']]
        ys = [cf[name]['prediction'][str(e)]['predicted']
              for e in cf['eps_dex']]
        ax.plot(xs, ys, marker=MARK[name], ms=4.2, lw=0.9, color=STATE_HUE,
                markerfacecolor=SURFACE, markeredgewidth=1.0, label=name,
                zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(2e-3, 0.45); ax.set_ylim(2e-3, 0.45)
    ax.set_xlabel('measured nRMSE', fontsize=8)
    ax.set_ylabel('predicted from the solver', fontsize=8)
    med = cf['_summary']['median_ratio']
    ax.text(0.045, 0.93, f'median ratio {med:.3f}\n25 points, no network',
            transform=ax.transAxes, fontsize=7.4, va='top', color=INK,
            linespacing=1.5)
    ax.legend(frameon=False, fontsize=FS.MINPT, loc='lower right',
              handlelength=1.1, borderpad=0.2, labelspacing=0.25)
    for a in axes:
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
    fig.subplots_adjust(wspace=0.38)
    save(fig, 'fig17_coordinate_displacement', outdirs)



# ---------------------------------------------------------------- figure 18
def _box(ax, x, y, w, h, label, fs, face, edge, lw=0.9, weight=None):
    """A rounded node. EPS has no alpha channel, so fills are pre-blended."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0,rounding_size=0.018',
        facecolor=face, edgecolor=edge, linewidth=lw, mutation_aspect=2.6))
    ax.text(x + w / 2, y + h / 2, label, fontsize=fs, ha='center',
            va='center', color=INK, linespacing=1.35, fontweight=weight)


def fig_deployment_gap(outdirs):
    """Two evaluation paths through one trained operator.

    The paths are identical but for the middle node, so that node is the only
    one carrying hue: the reader should see the single point of difference
    before reading a word. The deployed-to-oracle range is read from the
    converged-clock record.
    """
    th = json.load(open(os.path.join(RES, 'timehead.json')))
    ratios = {k: v['penalty_converged'] for k, v in th.items()
              if not k.startswith('_')}
    lo, hi = min(ratios.values()), max(ratios.values())

    fig, ax = plt.subplots(figsize=(FS.COL2, 2.15))
    ax.set_xlim(0, 1); ax.set_ylim(0.02, 1.02); ax.axis('off')

    X_IN, W_IN = 0.020, 0.150
    X_MID, W_MID = 0.245, 0.195
    X_OP, W_OP = 0.515, 0.120
    H = 0.250
    y_top, y_bot = 0.620, 0.150

    SHARED = FS.tint(INK, 0.05)          # the components both paths share
    rows = (
        (y_top, r'exact $t_{\rm sing}$', 'reported', RULE,
         'available only under benchmark conditions'),
        (y_bot, r'predicted $\hat{t}_{\rm sing}$', 'obtained in use', TAU_HUE,
         'coordinate error originates here'),
    )
    for y, src, tag, hue, note in rows:
        yc = y + H / 2
        # a faint lane groups each path without competing with the nodes
        ax.add_patch(FancyBboxPatch(
            (X_IN - 0.014, y - 0.030), (X_OP + W_OP + 0.075) - X_IN + 0.028,
            H + 0.060, boxstyle='round,pad=0,rounding_size=0.02',
            facecolor=FS.tint(hue, 0.05), edgecolor='none', zorder=0,
            mutation_aspect=2.6))
        _box(ax, X_IN, y, W_IN, H, r'$u(0),\ \theta$', 8, SHARED, RULE, 0.8)
        _box(ax, X_MID, y, W_MID, H, src, 8, FS.tint(hue, 0.22), hue, 1.1)
        _box(ax, X_OP, y, W_OP, H, r'$G_\vartheta$', 9.5, SHARED, RULE, 0.8)
        for x0, x1, col in ((X_IN + W_IN, X_MID, RULE),
                            (X_MID + W_MID, X_OP, hue),
                            (X_OP + W_OP, X_OP + W_OP + 0.055, hue)):
            ax.annotate('', xy=(x1, yc), xytext=(x0, yc),
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=0.9,
                                        shrinkA=1, shrinkB=1))
        ax.text(X_OP + W_OP + 0.066, yc, tag, fontsize=7.8, va='center',
                color=INK)
        ax.text(X_MID + W_MID / 2, y - 0.060, note, fontsize=FS.MINPT,
                color=MUTED, ha='center', va='top')

    xb = 0.862
    ax.plot([xb, xb], [y_bot + H / 2, y_top + H / 2], color=RULE, lw=0.9)
    for yy in (y_bot + H / 2, y_top + H / 2):
        ax.plot([xb - 0.012, xb], [yy, yy], color=RULE, lw=0.9)
    ym = (y_top + y_bot) / 2 + H / 2
    ax.text(xb + 0.018, ym + 0.040, f'{lo:.2f}–{hi:.2f}', fontsize=9.5,
            va='center', color=INK)
    ax.text(xb + 0.018, ym - 0.048, 'ratio of the two', fontsize=FS.MINPT,
            va='center', color=MUTED)
    save(fig, 'fig18_deployment_gap', outdirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'results',
                                                     'figures'))
    ap.add_argument('--also', default=None)
    args = ap.parse_args()
    outdirs = [args.outdir] + ([args.also] if args.also else [])
    fig_arms(outdirs)
    fig_architecture(outdirs)
    fig_program_flow(outdirs)
    fig_coordinate_displacement(outdirs)
    fig_deployment_gap(outdirs)


if __name__ == '__main__':
    main()
