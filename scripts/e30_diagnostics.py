"""Diagnostics for the E30 Concerto structural test (no figure existed for it).

Three things the E30 artefact records but its summary never examines:

  (a) the no-evolution null is NESTED in the model -- the track starts at the
      CDM initial condition, so phase 0 IS the null, and `frac_better_than_null`
      cannot come out below 1;
  (b) the residual GROWS with scattering strength, i.e. the agreement is best
      where SIDM physics does least;
  (c) the fitted phase does not track Concerto's own parametric phase `tr1`.

Output: results/figures/e30_diagnostics.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), '..')
BLUE, ORANGE, INK, MUTED = '#0072B2', '#D55E00', '#1a1a1a', '#6b6b6b'

R = json.load(open(os.path.join(ROOT, 'results', 'e30', 'concerto.json')))['rows']
ph = np.array([r['best_phase'] for r in R])
rd = np.array([r['resid_dex'] for r in R])
nd = np.array([r['null_resid_dex'] for r in R])
sh = np.array([r['sigma_hat'] for r in R])
tr = np.array([r['tr1_theirs'] for r in R])
z = ph < 1e-3                      # pinned at phase 0 == the null

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.3))
for a in ax:
    a.grid(True, lw=0.5, color='#e3e3e3', zorder=0)
    a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        a.spines[s].set_color(MUTED)
    a.tick_params(colors=MUTED, labelsize=9)

# (a) nesting: every point on or below y = x, by construction
lim = [4e-3, 1.2]
ax[0].plot(lim, lim, color=MUTED, lw=1.2, ls='--', zorder=1,
           label='model = null (y = x)')
ax[0].scatter(nd[~z], rd[~z], s=42, marker='o', facecolor=BLUE,
              edgecolor='white', linewidth=0.8, zorder=3,
              label=f'phase fitted, n={int((~z).sum())}')
ax[0].scatter(nd[z], rd[z], s=62, marker='^', facecolor=ORANGE,
              edgecolor='white', linewidth=0.8, zorder=4,
              label=f'pinned at phase 0, n={int(z.sum())}')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('no-evolution null residual (dex)', fontsize=10, color=INK)
ax[0].set_ylabel('fitted-track residual (dex)', fontsize=10, color=INK)
ax[0].set_title('(a) the null is nested in the model', fontsize=11, color=INK, loc='left')
ax[0].legend(fontsize=8, frameon=False, loc='upper left')

# (b) relative skill vs scattering strength.
# The ABSOLUTE residual does rise with sigma_hat (MWU p=0.0006 across the
# median split), but so does the null: stronger scattering means more evolution
# and so more room to be wrong. The scale-free measure is resid/null, and that
# is FLAT -- the model holds ~10% of the null at every scattering strength.
# Plotting the absolute residual here instead reads as "worse where the physics
# matters", which the data do not support.
frac = rd / nd
rho_b, p_b = spearmanr(sh[~z], frac[~z])
ax[1].axhline(1.0, color=MUTED, lw=1.2, ls='--', zorder=1, label='no better than null')
ax[1].scatter(sh[~z], frac[~z], s=42, marker='o', facecolor=BLUE,
              edgecolor='white', linewidth=0.8, zorder=3)
ax[1].scatter(sh[z], frac[z], s=62, marker='^', facecolor=ORANGE,
              edgecolor='white', linewidth=0.8, zorder=4)
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlabel(r'$\hat{\sigma}$  (dimensionless cross-section)', fontsize=10, color=INK)
ax[1].set_ylabel('residual / null  (1 = no skill)', fontsize=10, color=INK)
ax[1].set_title('(b) relative skill is flat in scattering strength',
                fontsize=11, color=INK, loc='left')
ax[1].annotate(rf'fitted only: Spearman $\rho={rho_b:+.3f}$, $p={p_b:.2f}$',
               xy=(0.03, 0.06), xycoords='axes fraction', fontsize=9, color=INK)
ax[1].legend(fontsize=8, frameon=False, loc='upper left')

# (c) our fitted phase against Concerto's own parametric phase.
# Half their tr1 values sit exactly at the 1.1 ceiling, so the full-sample rank
# test is confounded by censoring. The honest statement is the uncensored one,
# and at n=41 it is an INCONCLUSIVE NULL -- no evidence of agreement, and not
# enough power to claim disagreement either. Normalisations also differ (ours is
# t/t_end, theirs t/t_c), so only the rank is comparable at all.
cen = tr >= 1.0999
rho_c, p_c = spearmanr(ph[~cen], tr[~cen])
ax[2].scatter(tr[~cen], ph[~cen], s=42, marker='o', facecolor=BLUE,
              edgecolor='white', linewidth=0.8, zorder=3,
              label=f'uncensored, n={int((~cen).sum())}')
ax[2].scatter(tr[cen], ph[cen], s=42, marker='s', facecolor='none',
              edgecolor=MUTED, linewidth=1.0, zorder=2,
              label=f'censored at $t_{{r1}}=1.1$, n={int(cen.sum())}')
ax[2].set_xlabel(r"Concerto parametric phase $t_{r1}$", fontsize=10, color=INK)
ax[2].set_ylabel('our fitted phase', fontsize=10, color=INK)
ax[2].set_title('(c) no detectable phase agreement', fontsize=11, color=INK, loc='left')
ax[2].annotate(rf'uncensored: $\rho={rho_c:+.3f}$, $p={p_c:.2f}$' + '\n(inconclusive at this n)',
               xy=(0.03, 0.80), xycoords='axes fraction', fontsize=9, color=INK)
ax[2].legend(fontsize=8, frameon=False, loc='center left')

fig.tight_layout()
out = os.path.join(ROOT, 'results', 'figures', 'e30_diagnostics.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=170, facecolor='white')
print('wrote', out)
print(f'(a) pinned {int(z.sum())}/{len(R)}; model never loses: {(rd > nd).sum()} halos')
print(f'(b) rho={rho_b:+.3f} p={p_b:.4f}')
print(f'(c) rho={rho_c:+.3f} p={p_c:.3f}')
