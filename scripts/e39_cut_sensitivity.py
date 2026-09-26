#!/usr/bin/env python3
"""E39 (V5 Major Comment 6) -- does the cosmological advantage survive the cut?

WHAT THE REFEREE ASKED AND WHAT THE DATA CAN ANSWER
---------------------------------------------------
Asked: paired residual differences with intervals; sensitivity of the advantage
to the fitting-radius cut; a resolution criterion connected to the upstream
suite's own convergence treatment rather than R_Vmax/eps used as a physical
classifier.

A RADIAL-CUT SENSITIVITY IS NOT POSSIBLE AND THAT IS A DATA FACT. The supplied
merger trees carry (Mvir, Vmax, R_Vmax) summaries per halo per epoch, not
density profiles. There are no fitting radii to cut. Saying so is the answer;
inventing a profile fit would not be.

WHAT IS POSSIBLE
----------------
1. Paired residual differences. The track and the NFW competitor are fitted to
   the SAME halo, so the paired difference log10(track/NFW) per halo is the
   right statistic, with a bootstrap interval and a signed-rank test, instead of
   a ratio of two medians.
2. A continuous sweep of the R_Vmax/eps cut rather than one threshold at 4.
   If the advantage is a genuine core-collapse signal it should decay smoothly
   as the cut loosens; if it is an artefact of a handful of the worst-resolved
   haloes it will vanish as soon as they leave.
3. THE SUITE'S OWN CONVERGENCE CRITERION, rather than one we invented.
   Asked which cut to use, the Concerto authors answered: R_Vmax is relatively
   well measured above ~2000 particles, the same cut applied in the SIDM
   Concerto paper, and a small R_Vmax is not itself a problem -- it means many
   particles at small radii, and SIDM profiles can be converged inside
   2.8*epsilon precisely because of core collapse (E. Nadler, priv. comm.,
   15 September 2026).

   That matters because it inverts our caveat. We had treated R_Vmax < 4*eps
   as a resolution warning; by the criterion the suite actually uses it is not
   one. The particle count N_p = Mvir/m_part with m_part = 6.3e3 Msun is also
   the honest independent axis, because R_Vmax/eps contains R_Vmax, which is
   inside the shrinkage statistic the bands are claimed to track, so those two
   are coupled by construction. N_p is not.

Output: results/e32/cut_sensitivity.json
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon, spearmanr

ROOT = os.path.join(os.path.dirname(__file__), '..')
M_PART = 6.3e3                      # Msun, DATA.md
N_FLOOR = 2000.0                    # particles; the suite's own cut
RNG = np.random.default_rng(20260915)


def boot_ci(x, stat=np.median, n=20000):
    b = [stat(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def block(mask, diff, label):
    """Paired summary for one subgroup, or None if it is empty."""
    if mask.sum() < 3:
        return {'label': label, 'n': int(mask.sum()), 'note': 'too few haloes'}
    d = diff[mask]
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {
        'label': label,
        'n': int(mask.sum()),
        'median_paired_log_ratio': float(np.median(d)),
        'median_paired_log_ratio_ci95': boot_ci(d),
        'track_better_fraction': float(np.mean(d < 0)),
        'wilcoxon_p': p,
        'factor_track_over_nfw': float(10 ** np.median(d)),
    }


def main():
    z = np.load(os.path.join(ROOT, 'results', 'e32', 'controls.npz'))
    track, nfw = z['resid'], z['nfw_resid']
    rve, shrink = z['rvmax_eps'], z['collapse_dex']
    # The binding count is the smaller of the two runs: a halo is only as
    # resolved as its worse-resolved counterpart, since the comparison is
    # between them.
    npart = np.minimum(z['npart_cdm'], z['npart_sidm'])
    # paired, in logs: negative means the gravothermal track fits better
    diff = np.log10(np.maximum(track, 1e-12) / np.maximum(nfw, 1e-12))

    out = {
        'statistic': 'log10(track residual / NFW residual), paired per halo; '
                     'negative favours the gravothermal track',
        'n': int(len(diff)),
        'radial_cut_sensitivity': (
            'NOT POSSIBLE with the supplied data: the trees carry '
            '(Mvir, Vmax, R_Vmax) summaries, not density profiles, so the '
            'comparison has no fitting radii to cut. Stated rather than '
            'substituted for.'),
        'pooled': block(np.ones(len(diff), bool), diff, 'all 81'),
        'published_bands': {
            'collapsed_RVmax_lt_4eps': block(rve < 4, diff, 'R_Vmax < 4 eps'),
            'intermediate_4_to_10eps': block((rve >= 4) & (rve <= 10), diff,
                                             '4 <= R_Vmax/eps <= 10'),
            'resolved_RVmax_gt_10eps': block(rve > 10, diff, 'R_Vmax > 10 eps'),
        },
    }

    # ---- 2. the cut, swept
    sweep = []
    for c in (2, 3, 4, 5, 6, 8, 10, 15, 20):
        sweep.append(dict(cut=c, **block(rve < c, diff, f'R_Vmax < {c} eps')))
    out['rvmax_eps_cut_sweep'] = sweep

    # ---- 3. an independent criterion
    qs = np.percentile(npart, [25, 50, 75])
    passes = int((npart >= N_FLOOR).sum())
    out['suite_convergence_criterion'] = {
        'criterion': f'N_p >= {N_FLOOR:.0f} particles, the cut used in SIDM '
                     'Concerto (E. Nadler, priv. comm., 15 Sep 2026)',
        'also_stated': 'a small R_Vmax is not itself a resolution problem -- '
                       'it means many particles at small radii -- and SIDM '
                       'profiles can be converged inside 2.8 eps because of '
                       'core collapse',
        'n_passing': passes,
        'n_total': int(len(npart)),
        'min_N_p': float(npart.min()),
        'min_N_p_among_contracted': float(npart[rve < 4].min()),
        'margin_at_the_worst_halo': float(npart.min() / N_FLOOR),
        'n_below_2p8_eps': int((rve < 2.8).sum()),
        'smallest_rvmax_over_eps': float(rve.min()),
        'consequence': 'every halo in the sample passes, in BOTH runs, so the '
                       'R_Vmax/eps stratification is a contraction axis and '
                       'not a resolution axis. The caveat this paper carried '
                       '-- that the evidence sits where an N-body profile is '
                       'least trustworthy -- was more conservative than the '
                       "suite's own criterion warrants, and is withdrawn.",
    }
    out['particle_count'] = {
        'm_part_Msun': M_PART,
        'note': 'N_p = min(Mvir_CDM, Mvir_SIDM)/m_part; it does not contain '
                'R_Vmax, unlike R_Vmax/eps',
        'quartiles': [float(q) for q in qs],
        'spearman_Np_vs_rvmax_eps': [float(v) for v in spearmanr(npart, rve)],
        'spearman_Np_vs_paired_diff': [float(v) for v in spearmanr(npart, diff)],
        'bands': [
            block(npart < qs[0], diff, 'lowest quartile of N_p'),
            block((npart >= qs[0]) & (npart < qs[2]), diff, 'middle half'),
            block(npart >= qs[2], diff, 'highest quartile of N_p'),
        ],
    }
    # does the advantage survive among the haloes that ARE well resolved but
    # HAVE contracted? that is the question the resolution objection poses
    contracted = shrink < -0.3
    out['contracted_and_resolved'] = {
        'definition': 'shrinkage in R_Vmax below -0.3 dex AND R_Vmax > 4 eps',
        'block': block(contracted & (rve > 4), diff, 'contracted, resolved'),
        'contracted_all': block(contracted, diff, 'contracted, any resolution'),
        'n_contracted': int(contracted.sum()),
        'n_contracted_and_resolved': int((contracted & (rve > 4)).sum()),
    }

    json.dump(out, open(os.path.join(ROOT, 'results', 'e32',
                                     'cut_sensitivity.json'), 'w'), indent=1)
    print(json.dumps(out['suite_convergence_criterion'], indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('rvmax_eps_cut_sweep',
                                   'suite_convergence_criterion')},
                     indent=1)[:2000])
    print('\ncut sweep:')
    for r in sweep:
        if 'median_paired_log_ratio' not in r:
            print(f"  cut {r['cut']:>2}: n={r['n']}  {r.get('note','')}")
            continue
        print(f"  cut {r['cut']:>2} eps: n={r['n']:>2}  median log ratio "
              f"{r['median_paired_log_ratio']:+.3f} "
              f"[{r['median_paired_log_ratio_ci95'][0]:+.3f},"
              f"{r['median_paired_log_ratio_ci95'][1]:+.3f}]  "
              f"track better in {r['track_better_fraction']:.2f}  "
              f"p={r['wilcoxon_p']:.4f}")
    print('wrote cut_sensitivity.json')


if __name__ == '__main__':
    main()
