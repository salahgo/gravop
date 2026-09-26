"""Check that the benchmark guide's numbers still match the artefacts.

WHY THIS EXISTS
---------------
`STATECOND_BENCHMARK.md` is the document a THIRD PARTY calibrates against. It
is the one file in this repository whose staleness costs somebody else their
time rather than ours, and it is the file least likely to be re-read when an
experiment moves a number: nothing in the training loop touches it.

It had drifted. On 2026-09-07 it still said arm D was the baseline to beat
(P20 made converged arm B stronger in three of five systems), still quoted the
P13 `A(0)` tolerances that P22 moved, still gave the branch accuracy from before
the unseeded-RNG fix, still quoted convergence margins scored against a 2x
reference that P21 showed is a third too generous, and had the endpoint
extrapolation cost as 10.8x where the artefact says 10.3x.

Every number the guide quotes is therefore read back and checked against the
record it comes from, and this check is a gate.

Usage:  check_benchmark_numbers.py [--doc STATECOND_BENCHMARK.md]
Exit:   non-zero if any check fails.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results', 'statecond')


def _load(name):
    return json.load(open(os.path.join(RES, name)))


def checks(doc):
    """(name, ok, detail) for every pinned number. `doc` is the raw text."""
    out = []

    def want(label, token):
        out.append((label, token in doc, f'expected "{token}"'))

    contract = _load('contract.json')
    sysrec = {k: v for k, v in contract.items() if not k.startswith('_')}

    # P22: the A(0) tolerances, against the BEST time arm and not against arm D
    for name, v in sysrec.items():
        tol = v['a0_tolerance_vs_best_time_dex']
        if tol is None:
            continue
        want(f'A(0) tolerance for {name}', f'{tol:.3f} dex')
    fails = sorted(k for k, v in sysrec.items()
                   if v['a0_tolerance_vs_best_time_dex'] is None)
    out.append(('systems with no workable A(0) tolerance are named',
                all(f'`{k}`' in doc for k in fails),
                f'must name {fails} as having no tolerance'))

    # P23: the branch classifier, after the unseeded-RNG fix
    acc = [v['branch_accuracy']['mean'] for v in sysrec.values()]
    slow = [v['branch_accuracy_slow_quintile']['mean']
            for v in sysrec.values()]
    want('branch accuracy range',
         f'{100 * min(acc):.1f}-{100 * max(acc):.1f}%')
    want('branch accuracy on the slow quintile',
         f'{100 * min(slow):.1f}-{100 * max(slow):.1f}%')

    # P20/P22: which arm is the strongest time-conditioned comparator, per
    # system. The guide must not tell a third party to beat arm D where arm B
    # is stronger.
    b_wins = sorted(k for k, v in sysrec.items()
                    if v['best_time_source'] == 'B_converged')
    d_wins = sorted(k for k, v in sysrec.items()
                    if v['best_time_source'] == 'D')
    out.append(('the systems where converged arm B beats arm D are named',
                all(f'`{k}`' in doc for k in b_wins),
                f'converged arm B is strongest in {b_wins}'))
    out.append(('the systems where arm D is strongest are named',
                all(f'`{k}`' in doc for k in d_wins),
                f'arm D is strongest in {d_wins}'))
    for name in b_wins:
        v = sysrec[name]
        want(f'converged arm B vs arm D in {name}',
             f"{v['armB_converged_reference']:.4f} vs "
             f"{v['arm_D_reference']:.4f}")

    # P23: the contract read with intervals -- the ties are the part a reader
    # will check, because the point estimates alone overstate the case
    # P22a (2026-09-12): read the BEST TIME ARM, not arm B. The best arm is D in
    # `film` and `burgers`, so `convB_over_armC_full` is the wrong comparator
    # there -- the same defect that reached the manuscript's Section 4.5.
    sig = _load('significance.json')['contract_ratios']
    key = 'bestTime_over_armC_full'
    for name, v in sig.items():
        r = v[key]
        want(f'contract ratio for {name}', f"{r['ratio']:.3f}")
        want(f'contract interval for {name}',
             f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    ties = sorted(k for k, v in sig.items() if v[key]['ci_contains_1'])
    out.append(('the systems where the contract comparison is a TIE are named '
                'as ties', all(f'`{k}`' in doc for k in ties)
                and 'tie' in doc.lower(), f'ties: {ties}'))

    # P21: convergence must be scored against a Richardson limit
    rr = _load('refined_reference.json')
    for name, v in rr.items():
        if name.startswith('_'):
            continue
        want(f'Richardson margin for {name}', f"{v['margin_richardson']:.1f}")
        want(f'hard-subset Richardson margin for {name}',
             f"{v['margin_hard_richardson']:.1f}")
    passing = rr['_summary']['passing']
    out.append(('only the systems that meet the convergence standard are '
                'claimed to', all(f'`{k}`' in doc for k in passing),
                f'passing: {passing}'))

    # P16: the endpoint policy, which a third party must reproduce or their
    # arm B is not comparable
    ep = _load('endpoint.json')
    fr = ep['_summary']['frac_queries_clipped_range']
    want('fraction of deployed queries clipped',
         f'{100 * min(fr):.1f}-{100 * max(fr):.0f}%')
    cost = [1 / v['clip_penalty_ratio'] for k, v in ep.items()
            if not k.startswith('_')]
    want('cost of free extrapolation',
         f'{min(cost):.1f}x to {max(cost):.1f}x')

    # P24/P29a: the choice of state coordinate, and the absence of a criterion
    aux = _load('auxcoord.json')
    for name in ('rd', 'ks', 'sidm', 'film', 'burgers'):
        want(f'arm E vs arm C in {name}', f"{aux[name]['E_over_C']:.2f}x")
    inj = _load('injectivity.json')
    out.append(('the injectivity criterion is reported as FAILED',
                inj['verdict'].startswith('FAIL')
                and 'failed its pre-registered robustness test' in doc,
                'the guide must not offer a criterion the artefact refutes'))

    # E23: the phase-noise reversal thresholds, which bound where the method
    # stops paying
    for name, v in sysrec.items():
        want(f'reversal threshold for {name}',
             f"{v['reversal_threshold_dex']:g}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--doc', default=os.path.join(
        ROOT, 'STATECOND_BENCHMARK.md'))
    args = ap.parse_args()
    doc = open(args.doc).read()
    res = checks(doc)
    bad = [r for r in res if not r[1]]
    for name, ok, detail in res:
        if not ok:
            print(f'FAIL  {name}: {detail}')
    print(f'{len(res) - len(bad)}/{len(res)} benchmark-guide checks pass')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
