"""The E2-E4 result: pooling hides the only place the gravothermal track wins.

Against a FAIR one-parameter competitor (NFW at fixed Mvir/Rvir, concentration
free) the track is indistinguishable on the pooled 81 haloes -- Wilcoxon
p = 0.41. Stratified by how well RVmax is resolved, the picture inverts: the
track beats NFW by 4x among the most collapsed haloes and loses among the least
evolved ones, which are 51 of 81 and dominate any pooled statistic.

The tension this exposes is the honest headline: the haloes that carry the
evidence are the ones sitting closest to the softening length.

Output: results/figures/e32_controls.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
INK, MUTED = '#1a1a1a', '#6b6b6b'

z = np.load(os.path.join(ROOT, 'results', 'e32', 'controls.npz'))
d = json.load(open(os.path.join(ROOT, 'results', 'e32', 'controls.json')))
tr, nf, re_ = z['resid'], z['nfw_resid'], z['rvmax_eps']
sg, ph = z['sigma_hat'], z['phase']

bands = [(0, 4, ORANGE, 'o', r'$R_{V\max}<4\epsilon$'),
         (4, 10, GREEN, 's', r'$4\epsilon\!-\!10\epsilon$'),
         (10, 1e9, BLUE, '^', r'$>10\epsilon$')]

fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.4))
for a in ax:
    a.grid(True, lw=0.5, color='#e3e3e3', zorder=0); a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        a.spines[s].set_color(MUTED)
    a.tick_params(colors=MUTED, labelsize=9)

# (a) head to head
lim = [3e-3, 2.0]
ax[0].plot(lim, lim, ls='--', lw=1.2, color=MUTED, zorder=1)
ax[0].annotate('track wins below', xy=(0.045, 0.006), fontsize=8.5, color=MUTED)
for lo, hi, c, mk, lab in bands:
    m = (re_ >= lo) & (re_ < hi)
    ax[0].scatter(nf[m], tr[m], s=46, marker=mk, facecolor=c, edgecolor='white',
                  linewidth=0.8, zorder=3, label=f'{lab}, n={int(m.sum())}')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('NFW concentration baseline (dex)', fontsize=10, color=INK)
ax[0].set_ylabel('gravothermal track (dex)', fontsize=10, color=INK)
ax[0].set_title('(a) same free parameters, head to head', fontsize=11, color=INK, loc='left')
ax[0].legend(fontsize=8, frameon=False, loc='upper left')

# (b) the pooled result hides the split
lbl = [b['rvmax_over_eps'] for b in d['E4_resolution']['bands']]
lbl = [r'$<4\epsilon$', r'$4\!-\!10\epsilon$', r'$>10\epsilon$']
tm = [b['track_resid_median'] for b in d['E4_resolution']['bands']]
nm = [b['nfw_resid_median'] for b in d['E4_resolution']['bands']]
nn = [b['n'] for b in d['E4_resolution']['bands']]
x = np.arange(3); w = 0.36
ax[1].bar(x - w/2, tm, w, color=BLUE, label='gravothermal track', zorder=3)
ax[1].bar(x + w/2, nm, w, color=ORANGE, label='NFW baseline', zorder=3)
for i in range(3):
    ax[1].annotate(f'n={nn[i]}', xy=(x[i], max(tm[i], nm[i]) * 1.08),
                   ha='center', fontsize=8.5, color=MUTED)
ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, fontsize=9.5)
ax[1].set_yscale('log')
ax[1].set_xlabel(r'$R_{V\max}$ in units of softening $\epsilon$', fontsize=10, color=INK)
ax[1].set_ylabel('median residual (dex)', fontsize=10, color=INK)
ax[1].set_title('(b) pooled $p=0.41$ hides an inversion', fontsize=11, color=INK, loc='left')
ax[1].legend(fontsize=8.5, frameon=False, loc='upper left')

# (c) the bands ARE the collapse sequence, not a halo-size proxy.
# RVmax/eps correlates with the observed rmax shrinkage at rho = +0.815
# (p = 2e-20) but only rho = +0.42 with mass, and the bands do not differ
# significantly in mass (MWU p = 0.08). The <4eps band has median shrinkage
# -0.774 dex: a factor of six. So panel (b)'s inversion is physical -- the
# track wins where the halo actually collapsed -- and the cost of that is
# that those haloes sit closest to the softening.
shrink = z['collapse_dex']
for lo, hi, c, mk, lab in bands:
    m = (re_ >= lo) & (re_ < hi)
    ax[2].scatter(shrink[m], re_[m], s=46, marker=mk, facecolor=c,
                  edgecolor='white', linewidth=0.8, zorder=3)
ax[2].axhline(4, ls='--', lw=1.1, color=MUTED, zorder=1)
ax[2].axhline(10, ls=':', lw=1.1, color=MUTED, zorder=1)
ax[2].annotate(r'$4\epsilon$', xy=(-1.72, 4.3), fontsize=8.5, color=MUTED)
ax[2].annotate(r'$10\epsilon$', xy=(-1.72, 11), fontsize=8.5, color=MUTED)
ax[2].set_yscale('log')
ax[2].set_xlabel(r'observed collapse: $\log_{10}(R_{V\max}^{\rm SIDM}/R_{V\max}^{\rm CDM})$',
                 fontsize=9.5, color=INK)
ax[2].set_ylabel(r'$R_{V\max}/\epsilon$', fontsize=10, color=INK)
ax[2].set_title(r'(c) bands are collapse, not size ($\rho=+0.82$)',
                fontsize=11, color=INK, loc='left')

fig.tight_layout()
out = os.path.join(ROOT, 'results', 'figures', 'e32_controls.png')
fig.savefig(out, dpi=170, facecolor='white')
print('wrote', out)


def stratification_figure():
    """fig21, drawn at FINAL size (textwidth = 390pt = 5.40in), not scaled down.

    The analysis figure above is 13.4in wide; included at \\textwidth it would be
    scaled by 0.40 and its 9pt labels would render at 3.6pt, under Elsevier's 6pt
    secondary floor. Drawing at final size with scale 1 keeps every label at its
    stated size. Two panels rather than three: the third panel's content is the
    single number rho = +0.815, which the text already carries.
    """
    fig, ax = plt.subplots(1, 2, figsize=(5.40, 2.55))
    for a in ax:
        a.grid(True, lw=0.4, color='#e3e3e3', zorder=0); a.set_axisbelow(True)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            a.spines[s].set_color(MUTED)
        a.tick_params(colors=MUTED, labelsize=7)

    lim = [3e-3, 2.0]
    ax[0].plot(lim, lim, ls='--', lw=1.0, color=MUTED, zorder=1)
    for lo, hi, c, mk, lab in bands:
        m = (re_ >= lo) & (re_ < hi)
        ax[0].scatter(nf[m], tr[m], s=15, marker=mk, facecolor=c,
                      edgecolor='white', linewidth=0.4, zorder=3,
                      label=f'{lab} ({int(m.sum())})')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel('NFW baseline (dex)', fontsize=8, color=INK)
    ax[0].set_ylabel('gravothermal track (dex)', fontsize=8, color=INK)
    ax[0].set_title('(a) equal free parameters', fontsize=8.5, color=INK, loc='left')
    ax[0].legend(fontsize=6.2, frameon=False, loc='upper left', handletextpad=0.3)

    x = np.arange(3); w = 0.36
    ax[1].bar(x - w/2, tm, w, color=BLUE, label='track', zorder=3)
    ax[1].bar(x + w/2, nm, w, color=ORANGE, label='NFW', zorder=3)
    for i in range(3):
        ax[1].annotate(f'{nn[i]}', xy=(x[i], max(tm[i], nm[i]) * 1.10),
                       ha='center', fontsize=6.5, color=MUTED)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r'$<4\epsilon$', r'$4\!-\!10\epsilon$', r'$>10\epsilon$'],
                          fontsize=7.5)
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$R_{V\max}$ / softening', fontsize=8, color=INK)
    ax[1].set_ylabel('median residual (dex)', fontsize=8, color=INK)
    ax[1].set_title(r'(b) pooled $p=0.41$ hides this', fontsize=8.5, color=INK, loc='left')
    ax[1].legend(fontsize=6.5, frameon=False, loc='upper right', handletextpad=0.4)

    fig.tight_layout(pad=0.4)
    base = os.path.join(ROOT, 'results', 'figures',
                        'fig21_concerto_stratification')
    for ext in ('pdf', 'eps'):
        fig.savefig(base + '.' + ext, facecolor='white')
        print('wrote', base + '.' + ext)
    fig.savefig(os.path.join(ROOT, 'results', 'figures',
                             'fig21_concerto_stratification.png'), dpi=200,
                facecolor='white')


stratification_figure()
