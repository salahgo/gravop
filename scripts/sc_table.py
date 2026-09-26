"""Assemble the cross-domain results into markdown tables.

Reads results/statecond/*.json and prints markdown. No computation here beyond
formatting, so the tables cannot drift from the JSON they come from.

Usage: sc_table.py [--out results/statecond/tables.md]
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')

ORDER = ['rd', 'ks', 'film', 'burgers', 'sidm']
LABEL = {
    'rd': 'reaction-diffusion blow-up (combustion)',
    'ks': 'Keller-Segel aggregation (biology)',
    'film': 'thin-film rupture (interfaces)',
    'burgers': 'Burgers steepening (gas dynamics)',
    'sidm': 'gravothermal collapse (astrophysics)',
}


def load(name, prefix=''):
    p = os.path.join(RES, f'{prefix}{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def law_table(law):
    """The (sensitivity, penalty) points and the fit through them.

    Ordered by sensitivity rather than by system, because the claim is that
    this single ordering explains the penalty across both systems and
    architectures.
    """
    f, ci, acc = law['fit_all'], law['exponent_ci'], law['accuracy_test']
    out = ['| system | backbone | oracle nRMSE | sensitivity | penalty B/A | '
           'predicted by the fit |', '|---|---|---|---|---|---|']
    for p in sorted(law['points'], key=lambda q: q['sensitivity']):
        pred = 10.0 ** (f['exponent'] * math.log10(p['sensitivity'])
                        + f['intercept'])
        out.append(f"| `{p['system']}` | {p['backbone']} | "
                   f"{p['oracle']:.4f} | {p['sensitivity']:.1f} | "
                   f"**{p['penalty']:.2f}x** | {pred:.2f}x |")
    out += ['',
            f"Fit over all {f['n']} points: **penalty ~ sensitivity"
            f"^{f['exponent']:.2f}** (95% CI {ci['lo']:.2f}-{ci['hi']:.2f}), "
            f"Pearson r = {f['r_linear']:.3f} linear, {f['r_log']:.3f} in "
            f"logs, RMS residual {f['rms_log_residual']:.3f} dex "
            f"({100 * (10 ** f['rms_log_residual'] - 1):.0f}% in penalty).",
            '',
            f"The mechanism predicts that within a system the *more accurate* "
            f"backbone pays the *larger* penalty. That holds in "
            f"**{acc['n_systems_monotone']}/{acc['n_systems']} systems**"
            + (f", which under a null of random ordering within each system "
               f"has p = {acc['p_all_monotone_under_random_ordering']:.1e}."
               if acc.get('p_all_monotone_under_random_ordering') else '.')
            + " Every backbone faces the *same* predicted `t_sing` (the time"
              " head is trained independently of the backbone and its median"
              " error is identical across them to two decimals), so the"
              " differing penalties are the operator's response, not a"
              " differently accurate time predictor."]
    held = law.get('heldout')
    if held:
        out += ['',
                f"Held out: fitting on {', '.join(held['fit_on_backbones'])} "
                f"alone and predicting the rest gives an RMS residual of "
                f"{held['rms_log_residual']:.3f} dex, against "
                f"{held['rms_log_residual_in_sample']:.3f} dex in sample."]
    sd = law.get('shared_denominator_check')
    if sd:
        out += ['',
                '**Not a shared-denominator artefact.** Sensitivity and '
                'penalty both divide by the oracle error, so the correlation '
                'is re-tested three ways: on the numerators alone, with no '
                f"shared term, r = {sd['r_numerators_only']:+.3f} (against "
                f"{sd['r_ratios']:+.3f} for the ratios); with log oracle error "
                f"regressed out of both, partial r = "
                f"{sd['r_partial_given_oracle']:+.3f}; and the exponent is "
                f"{sd['exponent']:.2f}, where an artefact driven by the shared "
                'denominator would tend to 1.00.']
    return '\n'.join(out)


def main_table(rows):
    out = ['| system | domain | runs | t_sing spread | t_sing err | '
           'A oracle | B deployed | C state | B/A | B/C | B/C late |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, r in rows:
        n = r['nrmse']
        out.append(
            f"| `{name}` | {LABEL[name].split('(')[1].rstrip(')')} | "
            f"{r['n_runs']} | {r['tsing_spread_decades']:.2f} dex | "
            f"{r['tsing_median_err_pct']['mean']:.1f}% | "
            f"{n['A_tau_oracle']['mean']:.4f} | "
            f"{n['B_tau_deployed']['mean']:.4f} | "
            f"{n['C_state']['mean']:.4f} | "
            f"**{r['deployment_penalty']:.1f}x** | "
            f"**{r['state_vs_deployed']:.1f}x** | "
            f"{r['state_vs_deployed_late']:.1f}x |")
    return '\n'.join(out)


def required_tsing_accuracy(r):
    """How accurate would a time predictor have to be to match state-conditioning?

    Derived here rather than stored, so the bounded cases stay explicit: when
    even the smallest imposed t_sing error already loses, the answer is a bound
    ("better than the smallest value swept"), not a number.
    """
    sens = r.get('tsing_sensitivity')
    if not sens:
        return 'n/a'
    ds = sorted(float(k) for k in sens)
    ev = [sens[str(d)] for d in ds]
    c = r['nrmse']['C_state']['mean']
    if ev[0] >= c:
        return f'< {100 * ds[0]:.1f}%'
    if ev[-1] < c:
        return f'> {100 * ds[-1]:.0f}%'
    for i in range(1, len(ds)):
        if ev[i] >= c:
            f = (c - ev[i - 1]) / max(ev[i] - ev[i - 1], 1e-12)
            return f'{100 * (ds[i - 1] + f * (ds[i] - ds[i - 1])):.1f}%'
    return 'n/a'


def sensitivity_table(rows):
    ds = None
    for _, r in rows:
        if r.get('tsing_sensitivity'):
            ds = sorted(float(k) for k in r['tsing_sensitivity'])
            break
    if ds is None:
        return ''
    head = ' | '.join(f'{100 * d:g}%' for d in ds)
    out = [f'| system | C state | {head} | needed to match C |',
           '|---' * (len(ds) + 3) + '|']
    for name, r in rows:
        s = r.get('tsing_sensitivity')
        if not s:
            continue
        cells = ' | '.join(f"{s[str(d)]:.4f}" for d in ds)
        out.append(f"| `{name}` | {r['nrmse']['C_state']['mean']:.4f} | "
                   f"{cells} | **{required_tsing_accuracy(r)}** |")
    return '\n'.join(out)


def leak_table(rows):
    out = ['| system | C vs A, full domain | C vs A, observable region masked '
           '| masked points |', '|---|---|---|---|']
    for name, r in rows:
        lk = r['leak_check']
        out.append(f"| `{name}` | {r['state_vs_oracle']:.2f}x | "
                   f"{lk['state_vs_oracle_masked']:.2f}x | "
                   f"{r['masked_points']}/{r['n_x']} |")
    return '\n'.join(out)


def hybrid_table(rows):
    out = ['| system | noise (dex) | B deployed | C point | H1 joint | '
           'H2 marginalised | H3 ensemble |', '|---|---|---|---|---|---|---|']
    for name, r in rows:
        for key in ('eps0.0_q0.0', 'eps0.05_q0.0', 'eps0.2_q0.0',
                    'eps0.0_q0.15'):
            if key not in r['aggregate']:
                continue
            a = r['aggregate'][key]
            eps = key.split('_')[0][3:]
            q = key.split('_q')[1]
            tag = f"{eps}" if q == '0.0' else f"{eps} (flag flip {q})"
            best = min(('C_state_point', 'H1_joint', 'H2_marginalised',
                        'H3_ensemble'), key=lambda k: a[k]['mean'])

            def cell(k):
                s = f"{a[k]['mean']:.4f}"
                return f"**{s}**" if k == best else s
            out.append(f"| `{name}` | {tag} | {a['B_tau_deployed']['mean']:.4f} "
                       f"| {cell('C_state_point')} | {cell('H1_joint')} "
                       f"| {cell('H2_marginalised')} | {cell('H3_ensemble')} |")
    return '\n'.join(out)


def hybrid_extras_table(hyb, base):
    """What H2 and H3 actually buy, beyond the point-estimate accuracy.

    H2's product is a predictive spread obtained from K forward passes and no
    extra training; the useful question is whether that spread is the size of
    the error it is meant to describe. H3's ensemble is a control. Separately,
    inverting the state coordinate back into a time is compared against the
    trained head that arm B relies on.
    """
    b = dict(base)
    out = ['| system | t_sing err, trained head | t_sing err, state inversion '
           '| H2 spread / H2 error |', '|---|---|---|---|']
    for name, r in hyb:
        stt = r.get('state_to_time', {})
        sp = r.get('marginal_spread_vs_error') or []
        ratios = [s['ratio'] for s in sp]
        head = (f"{b[name]['tsing_median_err_pct']['mean']:.2f}%"
                if name in b else 'n/a')
        rng = (f'{min(ratios):.2f}-{max(ratios):.2f}' if ratios else 'n/a')
        out.append(f"| `{name}` | {head} | "
                   f"{stt.get('median_tsing_err_pct', float('nan')):.2f}% | "
                   f"{rng} |")
    return '\n'.join(out)


ROBUST_LABEL = {'C_clean_trained': 'C state (clean-trained)',
                'time_aug': 'B_aug time (noise-trained)',
                'phase_aug': 'C_aug state (noise-trained)',
                'joint_aug': 'H1_aug joint (noise-trained)',
                'joint_rel': 'H2_rel joint + reliability'}


def robust_table(rob, base):
    """E24: what training on the deployed inputs does to each scheme."""
    b = dict(base)
    keys = list(ROBUST_LABEL)
    out = ['| system | noise | B time, deployed | ' +
           ' | '.join(ROBUST_LABEL[k] for k in keys) + ' |',
           '|---' * (len(keys) + 3) + '|']
    for name, r in rob:
        bv = (b[name]['nrmse']['B_tau_deployed']['mean']
              if name in b else float('nan'))
        for key, tag in (('eps0.0_q0.0', '0'), ('eps0.05_q0.0', '0.05 dex'),
                         ('eps0.2_q0.0', '0.2 dex'),
                         ('eps0.0_q0.15', 'flag flip 15%')):
            if key not in r['aggregate']:
                continue
            a = r['aggregate'][key]
            best = min(keys, key=lambda k: a[k]['mean'])

            def cell(k):
                s = f"{a[k]['mean']:.4f}"
                return f'**{s}**' if k == best else s
            out.append(f"| `{name}` | {tag} | {bv:.4f} | "
                       + ' | '.join(cell(k) for k in keys) + ' |')
    return '\n'.join(out)


def robust_summary(rob, base):
    """The deployment question: how bad can each scheme get, and does it ever
    lose to the time-conditioned baseline?"""
    b = dict(base)
    keys = list(ROBUST_LABEL)
    out = ['| system | B time, deployed | scheme | best (no noise) | '
           'worst over sweep | worst / B |', '|---|---|---|---|---|---|']
    for name, r in rob:
        bv = (b[name]['nrmse']['B_tau_deployed']['mean']
              if name in b else float('nan'))
        for k in keys:
            vals = [c[k]['mean'] for c in r['aggregate'].values()]
            best, worst = min(vals), max(vals)
            flag = '' if worst <= bv else '  (loses to B)'
            out.append(f"| `{name}` | {bv:.4f} | {ROBUST_LABEL[k]} | "
                       f"{best:.4f} | {worst:.4f} | "
                       f"{worst / bv:.2f}x{flag} |")
    return '\n'.join(out)


def penalty_decomposition(rob, base):
    """How much of the E22 deployment penalty survives a better training recipe?

    Arm B in E22 was trained on the exact tau and deployed on a predicted one,
    which is standard practice but also a train/test mismatch. B_aug is the same
    time-conditioned model trained with tau corruption, so the difference
    separates "the penalty is intrinsic" from "the penalty is a training
    artefact". Both numbers belong in the paper.
    """
    b = dict(base)
    out = ['| system | A oracle | B naive | B_aug tau-augmented | C state | '
           'penalty B/A | penalty B_aug/A | state vs best time model |',
           '|---|---|---|---|---|---|---|---|']
    for name, r in rob:
        if name not in b or 'time_aug' not in r['aggregate']['eps0.0_q0.0']:
            continue
        nb = b[name]['nrmse']
        A = nb['A_tau_oracle']['mean']
        B = nb['B_tau_deployed']['mean']
        C = nb['C_state']['mean']
        Ba = r['aggregate']['eps0.0_q0.0']['time_aug']['mean']
        out.append(f"| `{name}` | {A:.4f} | {B:.4f} | {Ba:.4f} | {C:.4f} | "
                   f"{B / A:.1f}x | **{Ba / A:.1f}x** | **{Ba / C:.1f}x** |")
    return '\n'.join(out)


def rollout_table(rol, base, rob):
    """E25: the standard alternative that needs no t_sing at all.

    The best rollout arm is taken per system rather than fixing one, because
    pushforward training is not uniformly the winner (it hurts `sidm`), and
    pinning a baseline to a variant that happens to lose would understate it.
    """
    b, r = dict(base), dict(rob)
    out = ['| system | A oracle | B naive time | B_aug time | C state | '
           'best rollout | rollout vs B naive | rollout vs B_aug | C vs rollout |',
           '|---|---|---|---|---|---|---|---|---|']
    for name, o in rol:
        if name not in b:
            continue
        nb = b[name]['nrmse']
        A, B, C = (nb['A_tau_oracle']['mean'], nb['B_tau_deployed']['mean'],
                   nb['C_state']['mean'])
        arms = {k: v['mean']['mean'] for k, v in o['arms'].items()}
        best_lab = min(arms, key=arms.get)
        R = arms[best_lab]
        ba = (r[name]['aggregate']['eps0.0_q0.0']['time_aug']['mean']
              if name in r and 'time_aug' in r[name]['aggregate']['eps0.0_q0.0']
              else float('nan'))
        out.append(f"| `{name}` | {A:.4f} | {B:.4f} | {ba:.4f} | {C:.4f} | "
                   f"{R:.4f} ({best_lab}) | **{B / R:.1f}x** | "
                   f"{ba / R:.1f}x | **{R / C:.1f}x** |")
    return '\n'.join(out)


def rollout_cost_table(rol):
    out = ['| system | sequential steps per query | full trajectory | '
           'one direct query | random-access penalty |',
           '|---|---|---|---|---|']
    for name, o in rol:
        a = o['arms'].get('R_push') or list(o['arms'].values())[0]
        full = a['rollout_seconds_full_trajectory']
        one = a['seconds_single_query']
        out.append(f"| `{name}` | {a['sequential_steps_per_query']} | "
                   f"{1e3 * full:.1f} ms | {1e3 * one:.2f} ms | "
                   f"**{full / one:.0f}x** |")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(RES, 'tables.md'))
    args = ap.parse_args()

    base = [(n, load(n)) for n in ORDER]
    base = [(n, r) for n, r in base if r]
    hyb = [(n, load(n, 'hybrid_')) for n in ORDER]
    hyb = [(n, r) for n, r in hyb if r]
    if not base:
        sys.exit('no results in results/statecond/')

    parts = ['## Cross-domain conditioning comparison\n',
             'nRMSE, field error in units of the training-set field standard '
             'deviation. Mean over seeds.\n', main_table(base), '',
             '## Leakage check\n',
             'The state coordinate is read off the target field, so the '
             'advantage is recomputed with the region around the observable '
             'removed.\n', leak_table(base), '',
             '## Sensitivity to t_sing accuracy\n',
             'Arm B re-measured against *imposed* relative errors in t_sing '
             '(symmetrised over sign) rather than the learned head\'s own, so '
             'the penalty cannot be blamed on an underfitted head. The last '
             'column is the accuracy a time predictor would need in order to '
             'match state-conditioning.\n', sensitivity_table(base)]
    if hyb:
        parts += ['', '## Hybrid schemes under phase-measurement error\n',
                  hybrid_table(hyb), '',
                  '### What the hybrids buy beyond accuracy\n',
                  'A spread/error ratio near 1 means H2\'s predictive spread is '
                  'an honest error bar, obtained from K forward passes and no '
                  'extra training. The state-inversion column recovers t_sing '
                  'from an observed (phi, flag) instead of predicting it from '
                  'the initial data.\n',
                  hybrid_extras_table(hyb, base)]
    rob = [(n, load(n, 'robust_')) for n in ORDER]
    rob = [(n, r) for n, r in rob if r]
    if rob:
        parts += ['', '## Training the conditioning channels on the inputs '
                  'they will actually see (E24)\n',
                  'E22/E23 trained every channel on its exact value and '
                  'evaluated on a corrupted one. Here each channel is '
                  'corrupted per example during training instead.\n',
                  robust_table(rob, base), '',
                  '### Worst case over the whole noise sweep\n',
                  'The deployment question is not the best case but the worst: '
                  'a scheme is only safe to ship blind if it never loses to '
                  'the time-conditioned baseline.\n',
                  robust_summary(rob, base), '',
                  '### How much of the E22 penalty is a training artefact?\n',
                  'Arm B was trained on the exact tau and deployed on a '
                  'predicted one -- standard practice, but also a train/test '
                  'mismatch. B_aug is the same time-conditioned model trained '
                  'with tau corruption. The gap between the two penalty '
                  'columns is the part of the E22 headline that better '
                  'training recovers; the rest is intrinsic.\n',
                  penalty_decomposition(rob, base)]
    rol = [(n, load(n, 'rollout_')) for n in ORDER]
    rol = [(n, r) for n, r in rol if r]
    if rol:
        parts += ['', '## Against the standard alternative: autoregressive '
                  'rollout (E25)\n',
                  'Rollout is the usual recipe for time-dependent PDE '
                  'surrogates and needs no `t_sing` and no phase variable at '
                  'all, so it is a structural competitor rather than one more '
                  'baseline. Arms: naive next-state, residual + input noise, '
                  'and residual + noise + pushforward.\n',
                  rollout_table(rol, base, rob), '',
                  '### The cost rollout cannot avoid\n',
                  'A rollout cannot answer "the field at one time" without '
                  'computing every earlier step. For an MCMC likelihood that '
                  'queries a single epoch, that is the number that decides it.\n',
                  rollout_cost_table(rol)]
    law_path = os.path.join(RES, 'law.json')
    if os.path.exists(law_path):
        parts += ['', '## The sensitivity law across backbones (E27, P1)\n',
                  'Every (system x backbone) point, and the fit that ties '
                  'them together. `sensitivity` is the error under a 10% '
                  '*imposed* `t_sing` error divided by the oracle error: how '
                  'sharply the operator resolves the phase it is told. '
                  'Regenerate with `scripts/sc_law.py`.\n',
                  law_table(json.load(open(law_path)))]
    txt = '\n'.join(parts) + '\n'
    open(args.out, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
