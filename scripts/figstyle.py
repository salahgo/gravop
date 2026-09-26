"""One figure style for the whole manuscript, with the colour part computed.

WHY THIS EXISTS
---------------
Figure style was defined twice -- once in `sc_figures.py` for the thirteen
plotted figures, once in `sc_schematics.py` for the four schematics -- and the
two had drifted apart in ways that matter for the submission:

* `sc_figures.py` set no `ps.fonttype`/`pdf.fonttype`, so its output carried
  Type 3 fonts. Elsevier asks for vector artwork with all fonts embedded.
* `sc_figures.py` wrote only PDF and PNG. **CPC asks for EPS**, so thirteen of
  the seventeen figures had no EPS at all.
* Figure widths ranged 6.8-7.8 in. Elsevier's column widths are 90 mm and
  190 mm (3.543 in and 7.480 in); anything wider is scaled down in production,
  which shrinks the lettering below the 7 pt floor.

PALETTE
-------
Four categorical slots carrying arm identity, consistent across every figure.
Validated with the dataviz validator against the light surface on 2026-09-12:

    lightness band   PASS      chroma floor     PASS
    CVD separation   PASS  worst adjacent green<->blue dE 19.9 deutan
    normal vision    PASS  worst adjacent dE 20.9
    contrast >= 3:1  PASS  all four

The green moved from #1baf7a to #128a5e. The old value sat at 2.74:1 against
the surface, which the validator flags as a WARN that must be discharged by
visible labels; the new one clears the check outright, so the figures no longer
depend on every mark being labelled to stay legible.

Identity is never carried by colour alone: arms also differ in marker, and the
deployability glyphs are shape plus colour.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

# ---------------------------------------------------------------- palette
TAU_HUE = '#eb6834'      # time-conditioned arms (A, B)
STATE_HUE = '#2a78d6'    # state-conditioned arm (C)
DIRECT_HUE = '#128a5e'   # direct physical time (D)
AUX_HUE = '#7b5cd6'      # auxiliary coordinate (E)
CATEGORICAL = (TAU_HUE, STATE_HUE, DIRECT_HUE, AUX_HUE)

INK, MUTED, RULE = '#1a1a1a', '#5b5b5b', '#9a9a9a'
# CPC requires a plain white figure background: no off-white, gradient or tint.
SURFACE = '#ffffff'
GOOD, BAD = '#128a5e', '#c0392b'

BACKBONE_HUE = {'fno': STATE_HUE, 'deeponet': TAU_HUE,
                'cno': DIRECT_HUE, 'transformer': AUX_HUE}

# marker carries identity alongside hue, so the figures survive greyscale
MARK = {'rd': 'o', 'ks': 's', 'film': '^', 'burgers': 'D', 'sidm': 'P'}
SYS_ORDER = ['rd', 'ks', 'film', 'burgers', 'sidm']

# ------------------------------------------------------- Elsevier geometry
COL1 = 3.543    # 90 mm, single column
COL2 = 7.480    # 190 mm, double column

# Elsevier's artwork specification sets a 7 pt floor on lettering at the size
# the figure is printed. Every explicit fontsize in the figure scripts is at
# or above this; the rcParams below are too.
MINPT = 7.0

# Elsevier figures are set in a neutral grotesque. Helvetica and Arial are
# named first so a machine that has them uses them; DejaVu Sans is the portable
# fallback and embeds identically as Type 42.
_SANS = ['Helvetica', 'Arial', 'Nimbus Sans', 'Liberation Sans', 'DejaVu Sans']

BASE_RC = {
    'font.family': 'sans-serif', 'font.sans-serif': _SANS,
    'mathtext.fontset': 'dejavusans',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.edgecolor': INK, 'axes.linewidth': 0.6,
    'legend.frameon': False,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    # Type 42 embeds the glyphs. The default Type 3 is what production flags.
    'ps.fonttype': 42, 'pdf.fonttype': 42, 'ps.useafm': False,
    # EPS has no alpha channel: keep fills opaque rather than discovering a
    # rasterised patch in the proofs.
    'savefig.transparent': False,
}

PLOT_RC = dict(BASE_RC, **{
    'axes.spines.top': False, 'axes.spines.right': False,
    # A light grid, drawn opaque because PostScript has no alpha channel.
    'axes.grid': True, 'grid.alpha': 1.0, 'grid.linewidth': 0.35,
    'grid.color': '#e6e6e6', 'lines.linewidth': 1.3,
    # matplotlib's default is 'line', which puts the grid BEHIND lines but IN
    # FRONT OF patches, so an opaque grid draws pale stripes across every bar.
    # True puts it behind both.
    'axes.axisbelow': True,
    # short inward ticks on the two retained spines, no minor tick clutter
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.size': 0.0, 'ytick.minor.size': 0.0,
    'figure.dpi': 150,
})


def use(kind='plot'):
    """Install the shared style. `kind` is 'plot' or 'schematic'."""
    plt.rcParams.update(PLOT_RC if kind == 'plot' else BASE_RC)


def save(fig, name, outdirs, png=True):
    """Write EPS and PDF everywhere, and a PNG in the first dir for eyeballing.

    EPS is what CPC asks for; PDF keeps the pdflatex build from needing an
    epstopdf pass. Both come from the same vector render.
    """
    if isinstance(outdirs, str):
        outdirs = [outdirs]
    for d in outdirs:
        os.makedirs(d, exist_ok=True)
        for ext in ('eps', 'pdf'):
            fig.savefig(os.path.join(d, f'{name}.{ext}'), format=ext,
                        bbox_inches='tight', pad_inches=0.02)
    if png:
        fig.savefig(os.path.join(outdirs[0], f'{name}.png'), dpi=200,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', name)


def tint(hex_color, amount, surface=SURFACE):
    """Blend a hue toward the surface. EPS has no alpha, so tints are literal.

    `amount` is the weight on the hue: 0.12 gives a pale fill that a 1.3 pt
    stroke of the same hue reads cleanly against.
    """
    f = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(surface[i:i + 2], 16) for i in (1, 3, 5)]
    return '#%02x%02x%02x' % tuple(
        round(amount * x + (1 - amount) * y) for x, y in zip(f, b))
