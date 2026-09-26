"""Emit the benchmark's reference results in one machine-readable file.

A third party comparing their own architecture or their own system against this
benchmark should not have to parse markdown tables or open six JSONs. This
collapses the headline protocol results into `results/statecond/reference.json`
with the fields a comparison actually needs, and states the protocol alongside
them so the numbers cannot be quoted out of their conditions.

Usage: sc_reference.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
ORDER = ['rd', 'ks', 'film', 'burgers', 'sidm']
BACKBONES = {'': 'fno', '_deeponet': 'deeponet', '_cno': 'cno',
             '_transformer': 'transformer'}


def main():
    law = json.load(open(os.path.join(RES, 'law.json')))
    out = {
        'protocol': {
            'metric': 'field RMSE in units of the training-set field standard '
                      'deviation (nRMSE)',
            'arms': {
                'A_tau_oracle': 'conditioned on tau = t/t_sing with the TRUE '
                                't_sing supplied; not obtainable in use',
                'B_tau_deployed': 'conditioned on tau from a PREDICTED t_sing',
                'C_state': 'conditioned on (phi, branch flag) read off the '
                           'field; needs no t_sing',
            },
            'seeds': 3, 'epochs': 200,
            'split': 'by trajectory, 80/10/10, seed 1234',
            'late_cut': 0.92,
            'sensitivity': 'error under a 10% IMPOSED t_sing error, divided by '
                           'the oracle error',
            'reproduce': 'scripts/sc_gen.py --all --n 600 && '
                         'scripts/sc_experiment.py --all --backbone <name>',
        },
        'systems': {}, 'law': {}, 'points': [],
    }
    for s in ORDER:
        for suf, bk in BACKBONES.items():
            p = os.path.join(RES, f'{s}{suf}.json')
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            n = r['nrmse']
            out['points'].append({
                'system': s, 'backbone': r.get('backbone') or bk,
                'A_tau_oracle': n['A_tau_oracle']['mean'],
                'B_tau_deployed': n['B_tau_deployed']['mean'],
                'C_state': n['C_state']['mean'],
                'deployment_penalty_B_over_A': r['deployment_penalty'],
                'state_vs_deployed_B_over_C': r['state_vs_deployed'],
                'sensitivity_at_10pct': (r['tsing_sensitivity']['0.1']
                                         / n['A_tau_oracle']['mean']),
                'tsing_median_err_pct': r['tsing_median_err_pct']['mean'],
                'n_runs': r['n_runs'], 'n_test': r['n_test'],
            })
            if s not in out['systems']:
                out['systems'][s] = {
                    'n_runs': r['n_runs'], 'n_test': r['n_test'],
                    'n_fields': r['n_fields'], 'n_x': r['n_x'],
                    'tsing_spread_decades': r['tsing_spread_decades'],
                }
    f = law['fit_all']
    out['law'] = {
        'form': 'deployment_penalty ~ sensitivity ** exponent',
        'exponent': f['exponent'], 'intercept': f['intercept'],
        'r_log': f['r_log'], 'n_points': f['n'],
        'exponent_ci95': [law['exponent_ci']['lo'], law['exponent_ci']['hi']],
        'within_system_ordering': {
            'monotone_systems': law['accuracy_test']['n_systems_monotone'],
            'n_systems': law['accuracy_test']['n_systems'],
            'p_value': law['accuracy_test'][
                'p_all_monotone_under_random_ordering'],
        },
    }
    p = os.path.join(RES, 'reference.json')
    json.dump(out, open(p, 'w'), indent=1)
    print(f"wrote {os.path.relpath(p, ROOT)}: {len(out['points'])} points, "
          f"{len(out['systems'])} systems, "
          f"{len({q['backbone'] for q in out['points']})} backbones")


if __name__ == '__main__':
    main()
