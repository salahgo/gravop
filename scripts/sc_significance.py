"""E26 — how much of the cross-domain result is sampling noise?

An earlier assessment noted that the E22-E24 numbers rest on 3 seeds and ~60 test
trajectories per system, with seed spread reported but sampling error not. The
large effects are not in doubt; every narrow margin in the tables is currently
unsupported. This script supplies the missing statistics.

Two sources of variation are separated because they answer different questions:

  seed spread          how much the result depends on initialisation. Already
                       reported as `std` in the result files; repeated here for
                       completeness.
  trajectory bootstrap how much it depends on *which* test trajectories were
                       drawn. Resampled with replacement over test
                       trajectories, which is the error bar a referee means by
                       "is 60 trajectories enough?".

Ratios between arms are bootstrapped as *paired* quantities -- the same
resampled trajectory set is used for numerator and denominator -- because the
arms are evaluated on identical data and their errors are strongly correlated.
Treating them as independent would badly overstate the uncertainty on a ratio.

E22 now stores per-trajectory errors, so its headline ratios get a genuine
trajectory-level bootstrap. E23/E24 store per-seed values only, so their arm
comparisons are exact paired tests over seeds but carry no trajectory-level
interval; those are marked "(seeds)" in the output and are necessarily narrower
than the truth.

Usage:  sc_significance.py
Output: results/statecond/significance.json
"""
import itertools
import json
import os

import hashlib

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')
ORDER = ['rd', 'ks', 'film', 'burgers', 'sidm']
N_BOOT = 10000
RNG = np.random.default_rng(20260830)

# A single shared stream makes every published interval depend on how many
# comparisons were drawn before it, so ADDING a comparison silently perturbs
# the ones already in print. Adding the inferred-branch chain on 2026-09-14
# moved three intervals in their third decimal for exactly that reason.
# Seeding per comparison from its own name makes each interval a function of
# its own data alone.
_BASE_SEED = 20260830


def stream(*arrays):
    """A deterministic generator seeded from the comparison's own data.

    Keying on the data rather than on call order means an interval depends on
    nothing but the numbers it summarises, so adding a comparison cannot move
    one already in print.
    """
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return np.random.default_rng(_BASE_SEED + int.from_bytes(h.digest()[:4],
                                                             'big'))


def load(name, prefix=''):
    p = os.path.join(RES, f'{prefix}{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def paired_test(a, b):
    """Exact-ish paired comparison of two arms over shared seeds.

    n is 3, so this reports the paired mean difference, its spread, and a
    sign-consistency statement rather than a t-test p-value, which would be
    meaningless at this n. A difference is called consistent only if it has the
    same sign in every seed AND exceeds twice the paired standard error.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
    same_sign = bool(np.all(d > 0) or np.all(d < 0))
    return {'mean_diff': float(d.mean()), 'paired_se': float(se),
            'all_seeds_same_sign': same_sign,
            'consistent': bool(same_sign and abs(d.mean()) > 2 * se),
            'ratio': float(a.mean() / b.mean()) if b.mean() else float('nan')}


def bootstrap_ratio(num, den, n_traj):
    """Paired bootstrap of a ratio of means, resampling SEEDS.

    Kept for arms whose per-trajectory errors are not stored. Where they are,
    `bootstrap_traj` below is the one to use: this version cannot see test-set
    sampling error at all and its intervals are therefore too narrow.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    r = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(num), len(num))
        r.append(num[idx].mean() / den[idx].mean())
    r = np.array(r)
    return {'ratio': float(num.mean() / den.mean()),
            'ci_lo': float(np.percentile(r, 2.5)),
            'ci_hi': float(np.percentile(r, 97.5)),
            'resampled': 'seeds', 'n_test_trajectories': int(n_traj)}


def bootstrap_traj(per_num, per_den):
    """Paired bootstrap of a ratio of means, resampling TEST TRAJECTORIES.

    per_num/per_den are (n_seeds, n_traj) per-trajectory mean errors. The same
    resampled trajectory set is applied to numerator and denominator, and to
    every seed, because the arms are evaluated on identical trajectories and
    their errors are strongly correlated; resampling them independently would
    badly overstate the uncertainty on a ratio. Seeds are averaged within each
    resample, so this isolates test-set sampling error -- the question "is ~60
    trajectories enough?" -- rather than initialisation spread.
    """
    a = np.asarray(per_num, float)
    b = np.asarray(per_den, float)
    n = a.shape[1]
    rng = stream(a, b)
    r = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        r.append(a[:, idx].mean() / b[:, idx].mean())
    r = np.array(r)
    return {'ratio': float(a.mean() / b.mean()),
            'ci_lo': float(np.percentile(r, 2.5)),
            'ci_hi': float(np.percentile(r, 97.5)),
            'resampled': 'trajectories', 'n_test_trajectories': int(n)}


def main():
    out = {'n_boot': N_BOOT, 'note':
           'E22, and since v7/P23 also the timehead, contract and direct '
           'artefacts, are bootstrapped over test trajectories, paired. '
           'E23/E24 store per-seed means only, so their intervals resample '
           'seeds and are narrower than the truth; each entry records which '
           'was used in its "resampled" field.',
           'systems': {}}

    for name in ORDER:
        b = load(name)
        h = load(name, 'hybrid_')
        r = load(name, 'robust_')
        if not b:
            continue
        s = {'n_test': b['n_test']}

        # E22 arms: only aggregate mean/std over seeds are stored
        s['E22_seed_spread'] = {
            k: {'mean': v['mean'], 'std': v['std'],
                'rel_std': (v['std'] / v['mean']) if v['mean'] else None}
            for k, v in b['nrmse'].items()}

        # E23/E24 store per-seed rows, so paired comparisons are exact
        for tag, src, keys in (
                ('E23', h, ['B_tau_deployed', 'C_state_point', 'H1_joint',
                            'H2_marginalised', 'H3_ensemble']),
                ('E24', r, ['C_clean_trained', 'time_aug', 'phase_aug',
                            'joint_aug', 'joint_rel'])):
            if not src:
                continue
            cfgs = sorted({(x['eps_dex'], x['q']) for x in src['rows']})
            block = {}
            for eps, q in cfgs:
                rows = [x for x in src['rows']
                        if x['eps_dex'] == eps and x['q'] == q]
                have = [k for k in keys if k in rows[0]]
                per = {k: [x[k] for x in rows] for k in have}
                pairs = {}
                for a, bb in itertools.combinations(have, 2):
                    pairs[f'{a}_vs_{bb}'] = paired_test(per[a], per[bb])
                block[f'eps{eps}_q{q}'] = pairs
            s[f'{tag}_paired'] = block
        out['systems'][name] = s

    # headline ratios with seed-level CIs
    head = {}
    for name in ORDER:
        b, r = load(name), load(name, 'robust_')
        if not b:
            continue
        n_t = b['n_test']
        e = {}
        # E22's headline arms, resampled over test trajectories where the
        # per-trajectory errors were stored
        pt = b.get('per_trajectory')
        if pt:
            e['deployment_penalty_B_over_A'] = bootstrap_traj(
                pt['B_tau_deployed'], pt['A_tau_oracle'])
            e['state_vs_deployed_B_over_C'] = bootstrap_traj(
                pt['B_tau_deployed'], pt['C_state'])
            e['state_vs_oracle_A_over_C'] = bootstrap_traj(
                pt['A_tau_oracle'], pt['C_state'])
        if r:
            rows0 = [x for x in r['rows'] if x['eps_dex'] == 0.0 and x['q'] == 0.0]
            if rows0 and 'time_aug' in rows0[0]:
                e['state_vs_best_time'] = bootstrap_ratio(
                    [x['time_aug'] for x in rows0],
                    [x['C_clean_trained'] for x in rows0], n_t)
                e['joint_vs_best_time'] = bootstrap_ratio(
                    [x['time_aug'] for x in rows0],
                    [x['joint_aug'] for x in rows0], n_t)
        head[name] = e
    out['headline_ratios'] = head

    # ---- P23: the post-E22 experiments, now with trajectory-level CIs ------
    # Until v7 these carried a three-seed spread and nothing else, which is the
    # wrong error bar for the question. The converged deployment penalty is the
    # smallest effect in the project (0.97x in `burgers`, 1.06x in `sidm`), so
    # "is it different from 1?" is decided by test-set sampling error, not by
    # initialisation spread. All three artefacts now store per-trajectory
    # vectors on the same split, so the bootstrap is paired throughout.
    th = load('timehead')
    ct = load('contract')
    dr = load('direct')
    conv, contract_ci, direct_ci = {}, {}, {}
    for name in ORDER:
        if th and name in th and th[name].get('per_trajectory'):
            pt = th[name]['per_trajectory']
            conv[name] = {
                'penalty_converged_B_over_A': bootstrap_traj(pt['B'], pt['A']),
                'state_vs_deployed_converged_B_over_C':
                    bootstrap_traj(pt['B'], pt['C']),
                'state_vs_oracle_A_over_C': bootstrap_traj(pt['A'], pt['C']),
            }
            # the pre-registered question: does the factorisation still cost
            # anything once the clock is trained properly?
            for k, v in conv[name].items():
                v['ci_contains_1'] = bool(v['ci_lo'] <= 1.0 <= v['ci_hi'])
        if ct and name in ct and ct[name].get('per_trajectory'):
            pt = ct[name]['per_trajectory']
            blk = {
                'contract_cost_full_over_exact':
                    bootstrap_traj(pt['full'], pt['exact']),
                'aperture_over_exact':
                    bootstrap_traj(pt['aperture'], pt['exact']),
            }
            bt = ct[name].get('armB_converged_per_trajectory')
            if bt is not None:
                # > 1 means arm C under the full contract beats converged arm B
                blk['convB_over_armC_full'] = bootstrap_traj(bt, pt['full'])
                blk['convB_over_armC_exact'] = bootstrap_traj(bt, pt['exact'])
                for k in ('convB_over_armC_full', 'convB_over_armC_exact'):
                    blk[k]['ci_contains_1'] = bool(
                        blk[k]['ci_lo'] <= 1.0 <= blk[k]['ci_hi'])

            # The comparator the manuscript actually claims. P22 compares arm C
            # against min(arm B converged, arm D) PER SYSTEM, and for `film` and
            # `burgers` that is D, not B. Quoting `convB_over_armC_*` under a
            # "best time arm" heading -- which the reported tables did until
            # 2026-09-12 -- overstates arm C by 1.4x and 3.1x in those two.
            src = ct[name].get('best_time_source')
            best = None
            if src == 'B_converged':
                best = bt
            elif src == 'D' and dr and name in dr and dr[name].get('per_trajectory'):
                if dr[name].get('test_index') == ct[name].get('test_index'):
                    best = dr[name]['per_trajectory']['D_time_direct']
            if best is not None:
                # 'full' carries the ORACLE branch flag; 'branch_pred' replaces
                # it with the classifier, and is the only fully deployable
                # chain. The manuscript quoted intervals for 'full' alone,
                # which leaves an oracle inside a deployment claim.
                for suffix in ('full', 'exact', 'branch_pred'):
                    k = f'bestTime_over_armC_{suffix}'
                    blk[k] = bootstrap_traj(best, pt[suffix])
                    blk[k]['ci_contains_1'] = bool(
                        blk[k]['ci_lo'] <= 1.0 <= blk[k]['ci_hi'])
                    blk[k]['best_time_source'] = src
            contract_ci[name] = blk
        if dr and name in dr and dr[name].get('per_trajectory'):
            pt = dr[name]['per_trajectory']
            direct_ci[name] = {
                'D_over_C_state': bootstrap_traj(pt['D_time_direct'],
                                                 pt['C_state']),
                'D_over_A_oracle': bootstrap_traj(pt['D_time_direct'],
                                                  pt['A_tau_oracle']),
            }
            for v in direct_ci[name].values():
                v['ci_contains_1'] = bool(v['ci_lo'] <= 1.0 <= v['ci_hi'])
    out['converged_clock_ratios'] = conv
    out['contract_ratios'] = contract_ci
    out['direct_ratios'] = direct_ci

    path = os.path.join(RES, 'significance.json')
    json.dump(out, open(path, 'w'), indent=1)

    # The comparisons the papers actually claims, checked one by one. An
    # undifferentiated list of every pair is not useful: what matters is
    # whether each *stated* conclusion is supported by its own seeds.
    # `expect` is what we assert: 'gain' means the second arm should be better
    # (ratio > 1), 'null' means we assert no material difference. Reporting a
    # bare "significant/not" is ambiguous for a null claim, so each is judged
    # against its own direction.
    claims = [
        ('E24', 'eps0.0_q0.0', 'time_aug', 'C_clean_trained', 'gain',
         'state beats the BEST time model (headline)'),
        ('E24', 'eps0.0_q0.0', 'time_aug', 'joint_aug', 'gain',
         'fusion beats augmentation alone (the E24 control)'),
        ('E24', 'eps0.2_q0.0', 'C_clean_trained', 'joint_aug', 'gain',
         'joint is the hedge at high phase noise'),
        ('E24', 'eps0.0_q0.15', 'C_clean_trained', 'joint_aug', 'gain',
         'joint is the hedge under branch corruption'),
        ('E24', 'eps0.0_q0.0', 'joint_aug', 'joint_rel', 'null',
         'reliability inputs buy nothing material'),
        ('E23', 'eps0.2_q0.0', 'C_state_point', 'H2_marginalised', 'null',
         'marginalisation buys no accuracy'),
        ('E23', 'eps0.0_q0.0', 'C_state_point', 'H3_ensemble', 'gain',
         'ensembling helps when the phase is clean'),
    ]
    print('CLAIM CHECK — is each stated conclusion borne out by its own seeds?')
    print('  HOLDS / FAILS / unresolved; ratio > 1 favours the second arm.\n')
    for name, s in out['systems'].items():
        for tag, cfg, hi, lo, expect, desc in claims:
            # combinations() fixes the pair order, so look up both spellings
            # and invert the ratio when the stored pair is the other way round
            block = s.get(f'{tag}_paired', {}).get(cfg, {})
            st, inv = block.get(f'{hi}_vs_{lo}'), False
            if st is None:
                st, inv = block.get(f'{lo}_vs_{hi}'), True
            if st is None:
                continue
            ratio = 1.0 / st['ratio'] if inv and st['ratio'] else st['ratio']
            if expect == 'gain':
                verdict = ('HOLDS     ' if st['consistent'] and ratio > 1
                           else 'FAILS     ' if st['consistent']
                           else 'unresolved')
            else:                       # a null: material means > 5% and firm
                verdict = ('HOLDS     ' if abs(ratio - 1) < 0.05
                           or not st['consistent'] else 'FAILS     ')
            print(f"  {name:8s} {verdict} {ratio:6.2f}x  {desc}")
    print()

    # readable summary of every inconsistent (i.e. unsupported) comparison
    print('Comparisons that are NOT consistent across seeds '
          '(same sign in all 3 AND |diff| > 2 paired SE):\n')
    n_bad = 0
    for name, s in out['systems'].items():
        for tag in ('E23_paired', 'E24_paired'):
            for cfg, pairs in s.get(tag, {}).items():
                for pair, st in pairs.items():
                    if not st['consistent']:
                        n_bad += 1
                        print(f"  {name:8s} {tag[:3]} {cfg:16s} {pair:38s} "
                              f"ratio {st['ratio']:.3f}  "
                              f"diff {st['mean_diff']:+.5f} "
                              f"+-{st['paired_se']:.5f}")
    print(f"\n{n_bad} inconsistent comparisons.")
    print('\nHeadline ratios with 95% bootstrap CIs. "traj" resamples test\n'
          'trajectories and is the interval that answers "is ~60 enough?";\n'
          '"seeds" resamples initialisations only and is necessarily narrower.')
    for name, e in head.items():
        for k, v in e.items():
            print(f"  {name:8s} {k:28s} {v['ratio']:6.2f}x "
                  f"[{v['ci_lo']:5.2f}, {v['ci_hi']:5.2f}]  ({v['resampled']})")
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == '__main__':
    main()
