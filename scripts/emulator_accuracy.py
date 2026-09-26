"""One definition of "the emulator's accuracy", and the rest named.

An audit recorded that two aggregations of one metric were both quoted. The
real collision was subtler:

  * `results/evaluation/summary.json`, gravop_s0, test median relative density
    error = 1.3667%, quoted as "1.37%";
  * `results/e10/resolution.json` at 128 points, *mean* relative density error
    = 1.3736%, also quoted as "1.37%".

Two different statistics, on two different run sets (130-run test split vs the
60-run resolution subset), that happen to round to the same three characters.
Nothing flagged them because they agree to two decimals by coincidence.

THE CANONICAL DEFINITION, adopted here and to be used everywhere:

    the mean over the five production seeds of each seed's MEDIAN
    per-trajectory relative density error, on the 130-run test split,
    at oracle phase  =  1.39%

Why each part: the seed is this project's unit of replication, so
the outer aggregation is over seeds; the median is the inner one because the
mean is tail-dominated -- 2.02% against 1.39% for the same predictions, and
that gap is the tail, not a different model; and the test split is the
evaluation set every other headline uses.

Everything else is printed alongside so the alternatives keep their names and
nobody has to re-derive which is which.

The same script resolves the other standing collision (P6c): the
SIDM deployment penalty is 4.5x under E17 and 9.4x under E22. Both are correct
and they are not alternative aggregations of one measurement -- they are
different metrics, over different fields, on different populations. The
resolution is to state which everywhere, so the numbers are printed here with
their populations attached.

Usage:  emulator_accuracy.py
Output: results/evaluation/accuracy_definitions.json
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
EVAL = os.path.join(ROOT, 'results', 'evaluation')
SEEDS = range(5)


def penalty_block():
    """P6c: the two SIDM deployment penalties, each with its population.

    Neither supersedes the other. E17 asks "by how much does the shipped
    emulator's relative density error grow when it has to predict t_norm", E22
    asks the same question in the units the cross-domain protocol uses, over
    two fields, and only on the runs whose singularity is solution-dependent.
    """
    e17 = json.load(open(os.path.join(ROOT, 'results', 'e17',
                                      'error_budget.json')))
    sc = json.load(open(os.path.join(ROOT, 'results', 'statecond',
                                     'sidm.json')))
    return {
        'e17': {
            'value_x': float(e17['inflation_from_scalar_head']),
            'metric': 'mean per-trajectory relative density error (%)',
            'fields': ['density'],
            'population': ('130-run test split of the 1300-run '
                           'constant+velocity-dependent dataset'),
            'n_test': int(e17['n_test']),
            'oracle_pct': float(e17['A_oracle_tnorm']['mean_pct']),
            'deployed_pct': float(e17['B_predicted_tnorm']['mean_pct']),
        },
        'e22': {
            'value_x': float(sc['deployment_penalty']),
            'metric': 'nRMSE in units of the training-set field s.d.',
            'fields': ['density', 'velocity'],
            'population': ('61-run test split of the 603 solution-dependent '
                           'runs used by the cross-domain protocol'),
            'n_runs': int(sc['n_runs']), 'n_test': int(sc['n_test']),
            'oracle_nrmse': float(sc['nrmse']['A_tau_oracle']['mean']),
            'deployed_nrmse': float(sc['nrmse']['B_tau_deployed']['mean']),
        },
        'rule': ('always name the experiment and the population. STATE.md\'s '
                 'shorthand "all 1300 runs" and "the 603 runs" refers to the '
                 'dataset each test split is drawn from, not to the number of '
                 'runs measured, which is 130 and 61.'),
    }


def main():
    ev = json.load(open(os.path.join(EVAL, 'summary.json')))
    e10 = json.load(open(os.path.join(ROOT, 'results', 'e10',
                                      'resolution.json')))

    med = np.array([ev[f'gravop_s{s}']['test']['median_rel_rho']
                    for s in SEEDS])
    mean = np.array([ev[f'gravop_s{s}']['test']['mean_rel_rho']
                     for s in SEEDS])

    canonical = float(100 * med.mean())
    out = {
        'canonical': {
            'value_pct': canonical,
            'definition': ('mean over the 5 production seeds of each seed\'s '
                           'median per-trajectory relative density error, '
                           '130-run test split, oracle phase'),
            'seed_spread_pct': float(100 * med.std()),
            'per_seed_pct': (100 * med).tolist(),
            'quote_as': f'{canonical:.2f}%',
        },
        'alternatives': {
            'seed0_median_pct': float(100 * med[0]),
            'mean_over_seeds_of_seed_mean_pct': float(100 * mean.mean()),
            'per_seed_mean_pct': (100 * mean).tolist(),
            'e10_resolution_subset_mean_at_128_pct': float(
                100 * e10['per_resolution']['128']['mean_rel_rho']),
            'e10_resolution_subset_median_at_128_pct': float(
                100 * e10['per_resolution']['128']['median_rel_rho']),
            'e10_n_runs': int(e10['n_runs']),
        },
        'note': ('the mean is tail-dominated: 2.02% vs 1.39% on identical '
                 'predictions. Quote the median definition and say so, or '
                 'quote both, but never one as if it were the other.'),
        'audit_item_correction': ('A5 as recorded names 1.37% and 1.62%. '
                                  '1.62% does not appear in the manuscript or '
                                  'its git history; the actual collision is '
                                  'two distinct statistics that both round to '
                                  '1.37%.'),
        'deployment_penalty': penalty_block(),
    }
    json.dump(out, open(os.path.join(EVAL, 'accuracy_definitions.json'), 'w'),
              indent=1)

    c = out['canonical']
    print('CANONICAL emulator accuracy')
    print(f"  {c['value_pct']:.3f}%  +-{c['seed_spread_pct']:.3f} across seeds")
    print(f"  {c['definition']}")
    print(f"  per seed: {np.round(c['per_seed_pct'], 3).tolist()}")
    print('\nOTHER aggregations that have been quoted, with their real names')
    a = out['alternatives']
    print(f"  seed 0 median only ............................ "
          f"{a['seed0_median_pct']:.3f}%   (the rounded '1.37%')")
    print(f"  mean of per-seed MEANS (tail-dominated) ....... "
          f"{a['mean_over_seeds_of_seed_mean_pct']:.3f}%")
    print(f"  E10 resolution subset, mean at 128, n={a['e10_n_runs']} ...... "
          f"{a['e10_resolution_subset_mean_at_128_pct']:.3f}%"
          f"   (the other '1.37%')")
    print(f"  E10 resolution subset, median at 128 .......... "
          f"{a['e10_resolution_subset_median_at_128_pct']:.3f}%")
    print(f"\n{out['audit_item_correction']}")

    p = out['deployment_penalty']
    print('\nSIDM DEPLOYMENT PENALTY -- two correct numbers (P6c)')
    for tag in ('e17', 'e22'):
        b = p[tag]
        print(f"  {tag.upper()}  {b['value_x']:.2f}x  |  {b['metric']}  |  "
              f"{len(b['fields'])} field(s): {', '.join(b['fields'])}")
        print(f"        on {b['n_test']} test runs, {b['population']}")
    print(f"  {p['rule']}")
    print(f"\nwrote {os.path.join(EVAL, 'accuracy_definitions.json')}")


if __name__ == '__main__':
    main()
