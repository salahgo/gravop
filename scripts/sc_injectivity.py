"""P29a -- is the injectivity criterion a real ordering, or one statistic's artefact?

WHY THIS EXISTS
---------------
P24 (`sc_auxcoord.py`) found a SPLIT: conditioning the operator on a bulk L2
contrast `psi` instead of the singular observable `phi` is 18.7x worse in `rd`,
3.8x in `ks`, 3.3x in `sidm`, and *better* in `film` (0.88x) and `burgers`
(0.44x). That was reported with no account of why.

While planning this work one candidate account ordered the five systems
perfectly: how **non-monotone** the coordinate is along the trajectory. The
proposed criterion is that a state coordinate is worth conditioning on exactly
to the extent that it is INJECTIVE along the trajectory -- where it folds back,
two instants carry the same coordinate value and the operator is being asked an
ambiguous question. That is computable from the reference solutions alone, with
no network and no training, which is the same property that makes the
clock-error floor useful.

But `rho = 1.000` was obtained from ONE statistic ("fraction of sign-opposing
steps"), which is one arbitrary choice among several, at n = 5. This script
asks whether the ordering is a property of the coordinate or of that choice.

THE MEASURES (all computed on the stored reference solutions, no network)
-------------------------------------------------------------------------
  D0  nonmono_frac   fraction of steps whose d c/d tau sign opposes the
                     trajectory's net direction. THE ORIGINAL; carried here so
                     the scoping numbers are reproduced by a committed script.
  D1  tv_ratio       TV(c) / |c(end) - c(0)|, which is 1 for a monotone path
                     and grows with folding. Median over runs (the mean is
                     unbounded when the denominator is small).
  D2  collision_frac fraction of instant pairs (i, j) with |c_i - c_j| below 1%
                     of the run's range while |tau_i - tau_j| exceeds 20% of
                     the interval -- literal non-injectivity, at a tolerance.
  D3  inverse_resid  held-out RMSE of tau predicted from c alone by k-NN
                     regression, in units of tau. A direct measure of how much
                     the coordinate fails to determine the phase.
  X   range_ratio    dynamic range of psi relative to phi. THE CONTROL: it was
                     the other candidate while scoping and it did NOT order the
                     systems (rho = +0.200). If a measure only has to be
                     coordinate-shaped to correlate, this one would too.

PRE-REGISTRATION (written before D1-D3 were computed)
-----------------------------------------------------
Target: Spearman rho between the measure evaluated on `psi` and `E/C` from
`results/statecond/auxcoord.json`, over the five systems.

  PASS      all three of D1, D2, D3 give rho >= +0.9 (at most one adjacent
            transposition of the five). -> the criterion is a property of the
            coordinate, not of one statistic; P29b (the causal flag test) is
            justified and proceeds.
  PARTIAL   exactly two of three. -> report as partially robust, name the
            definition that fails and why, and proceed to P29b with the caveat
            stated wherever the criterion appears.
  FAIL      one or none. -> the criterion is an artefact of the sign-counting
            statistic. P29 STOPS HERE and is reported as a negative; P24 stays
            the descriptive split it already is, which needs no mechanism to be
            a result.

The control X is expected NOT to reach +0.9. If it does, the test has no
discriminating power at n = 5 and the whole line is reported as unresolvable at
this sample size.

n = 5 RULE. Every p-value here is the EXACT two-sided
permutation p over all 120 orderings. For rho = 1.000 that is p = 0.0167, and
0.0167 is the FLOOR at n = 5 -- a perfect ordering cannot do better. Say so
every time the number is quoted. The asymptotic Spearman p would read 0.000 and
is never used.

Because a perfect rank correlation at n = 5 is the strongest result available
and still weak, this script also resamples the DEPENDENT variable: `E/C` is
recomputed on paired bootstraps of the test trajectories (the per-trajectory
errors stored by `sc_auxcoord.py`), and the fraction of replicates in which
each measure still reaches rho >= 0.9 is reported. That separates "the ordering
is real" from "the ordering is one lucky draw of five point estimates".

Usage:
  sc_injectivity.py [--coords psi phi] [--boot 2000]
Output: results/statecond/injectivity.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import statecond.data as scd                                     # noqa: E402
from sc_provenance import stamp                                  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'sidm', 'film', 'burgers']

# ---------------------------------------------------------------- coordinates


def coord_psi(d):
    """log10(||u(t)||_2 / ||u(0)||_2), exactly as `sc_auxcoord.psi_table`."""
    u = np.asarray(d['u'], dtype=np.float64)
    nrm = np.sqrt((u ** 2).mean(axis=(2, 3)))
    return np.log10(np.maximum(nrm, 1e-300) / np.maximum(nrm[:, :1], 1e-300))


def coord_phi(d):
    """log10(A(t)/A(0)), the singular observable arm C is conditioned on."""
    return np.asarray(d['phi'], dtype=np.float64)


COORDS = {'psi': coord_psi, 'phi': coord_phi}

# ------------------------------------------------------------------- measures


def d0_nonmono(c, taus):
    """Fraction of steps opposing the trajectory's net direction."""
    s = np.sign(np.diff(c, axis=1))
    dom = np.sign(c[:, -1] - c[:, 0])[:, None]
    dom = np.where(dom == 0, 1.0, dom)
    per_run = ((s * dom) < 0).mean(axis=1)
    return per_run


def d1_tv_ratio(c, taus, floor=1e-12):
    """TV(c) / |net displacement|; 1 exactly when monotone."""
    tv = np.abs(np.diff(c, axis=1)).sum(axis=1)
    net = np.abs(c[:, -1] - c[:, 0])
    return tv / np.maximum(net, floor)


def d2_collisions(c, taus, val_tol=0.01, tau_sep=0.20):
    """Fraction of (i, j) pairs that collide in value but not in time."""
    dt = np.abs(taus[:, None] - taus[None, :])
    far = np.triu(dt > tau_sep * (taus[-1] - taus[0]), 1)
    n_pairs = int(far.sum())
    rng = np.ptp(c, axis=1)
    out = np.empty(c.shape[0])
    for r in range(c.shape[0]):
        tol = val_tol * max(rng[r], 1e-300)
        close = np.abs(c[r][:, None] - c[r][None, :]) < tol
        out[r] = (close & far).sum() / max(n_pairs, 1)
    return out


def d3_inverse_residual(c, taus, tr, te, k=10):
    """Held-out RMSE of tau predicted from c alone, by k-NN in c.

    Pooled over (run, tau) pairs: the training bank is every instant of every
    training run, the query set every instant of every test run. Nonparametric
    on purpose -- the question is how much the coordinate DETERMINES the phase,
    not whether some particular functional form fits it.
    """
    bank_c = c[tr].ravel()
    bank_t = np.tile(taus, len(tr))
    order = np.argsort(bank_c)
    bank_c, bank_t = bank_c[order], bank_t[order]
    q = c[te].ravel()
    truth = np.tile(taus, len(te))
    idx = np.searchsorted(bank_c, q)
    lo = np.clip(idx - k, 0, len(bank_c))
    hi = np.clip(idx + k, 0, len(bank_c))
    pred = np.empty_like(q)
    for i in range(len(q)):
        pred[i] = bank_t[lo[i]:hi[i]].mean() if hi[i] > lo[i] else 0.5
    return float(np.sqrt(((pred - truth) ** 2).mean()))


# ------------------------------------------------------------------ statistics


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else 0.0


def spearman_tiebreak(a, b, best):
    """rho after breaking ties in `a` in the order most (or least) favourable
    to `b`.

    D1 and D2 assign several systems an identical value -- a path that is
    exactly monotone has TV ratio exactly 1 and zero collisions -- so the
    average-rank convention above costs rho even when no system is INVERTED.
    Reporting both bounds keeps a verdict from turning on a coin flip.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    eps = 1e-9 * (np.ptp(a) if np.ptp(a) else 1.0)
    rb = rankdata(b)
    adj = a + (eps * rb if best else -eps * rb)
    return spearman(adj, b)


def exact_perm_p(a, b):
    """Two-sided exact permutation p for Spearman rho at small n."""
    rho = spearman(a, b)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cnt = tot = 0
    for perm in itertools.permutations(range(len(a))):
        tot += 1
        if abs(spearman(a[list(perm)], b)) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, cnt / tot, tot


# ----------------------------------------------------------------------- main


def within_traj_inverse_resid(c, taus):
    """POST HOC (not pre-registered, not counted in the verdict).

    D3 as specified fits tau from the coordinate POOLED over runs, so its
    residual mixes two things: folding within a trajectory, which is what the
    criterion is about, and spread BETWEEN trajectories at the same tau, which
    is not. This variant removes the second by asking the question one
    trajectory at a time: within a run, how well does the coordinate value
    determine the instant? For a strictly monotone path the answer is exactly,
    and the residual is 0 by construction.
    """
    out = []
    for r in range(c.shape[0]):
        v = c[r]
        rng = max(np.ptp(v), 1e-300)
        # nearest-value lookup, excluding the query instant itself
        dv = np.abs(v[:, None] - v[None, :]) / rng
        np.fill_diagonal(dv, np.inf)
        j = np.argmin(dv, axis=1)
        out.append(np.abs(taus[j] - taus))
    return float(np.mean(np.concatenate(out)))


def measures_for(name, coords):
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    taus = scd.tau_grid()
    tr, _, te = scd.split(d['u'].shape[0])
    rec = {'n_runs': int(d['u'].shape[0]), 'n_tau': int(len(taus))}
    for cname in coords:
        c = COORDS[cname](d)
        rec[cname] = {
            'D0_nonmono_frac': float(d0_nonmono(c, taus).mean()),
            'D1_tv_ratio_median': float(np.median(d1_tv_ratio(c, taus))),
            'D1_tv_ratio_mean': float(np.mean(d1_tv_ratio(c, taus))),
            'D2_collision_frac': float(d2_collisions(c, taus).mean()),
            'D3_inverse_resid': d3_inverse_residual(c, taus, tr, te),
            'range_dex': float(np.ptp(c)),
            'D3b_within_traj_resid_POSTHOC': within_traj_inverse_resid(c, taus),
        }
    if 'psi' in rec and 'phi' in rec:
        rec['X_range_ratio'] = (rec['psi']['range_dex']
                                / max(rec['phi']['range_dex'], 1e-300))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coords', nargs='+', default=['psi', 'phi'])
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260907)
    args = ap.parse_args()

    aux = json.load(open(os.path.join(OUT, 'auxcoord.json')))
    per_sys = {s: measures_for(s, args.coords) for s in SYSTEMS}
    ec = np.array([aux[s]['E_over_C'] for s in SYSTEMS])

    keys = ['D0_nonmono_frac', 'D1_tv_ratio_median', 'D2_collision_frac',
            'D3_inverse_resid']
    corr = {}
    for k in keys:
        v = np.array([per_sys[s]['psi'][k] for s in SYSTEMS])
        rho, p, tot = exact_perm_p(v, ec)
        corr[k] = {'values': v.tolist(), 'rho': rho, 'exact_p': p,
                   'n_perm': tot, 'passes_0p9': bool(rho >= 0.9),
                   'rho_tiebreak_best': spearman_tiebreak(v, ec, True),
                   'rho_tiebreak_worst': spearman_tiebreak(v, ec, False)}
    # the control, and the psi-minus-phi excess as a secondary reading
    v = np.array([per_sys[s]['X_range_ratio'] for s in SYSTEMS])
    rho, p, tot = exact_perm_p(v, ec)
    corr['X_range_ratio'] = {'values': v.tolist(), 'rho': rho, 'exact_p': p,
                             'n_perm': tot, 'passes_0p9': bool(rho >= 0.9),
                             'rho_tiebreak_best': spearman_tiebreak(v, ec, True),
                             'rho_tiebreak_worst': spearman_tiebreak(v, ec, False),
                             'role': 'CONTROL: expected NOT to order'}
    for k in keys:
        v = np.array([per_sys[s]['psi'][k] - per_sys[s]['phi'][k]
                      for s in SYSTEMS])
        rho, p, tot = exact_perm_p(v, ec)
        corr[k + '_excess_over_phi'] = {
            'values': v.tolist(), 'rho': rho, 'exact_p': p, 'n_perm': tot,
            'passes_0p9': bool(rho >= 0.9),
            'role': 'SECONDARY: psi measure minus the same measure on phi'}

    # bootstrap the DEPENDENT variable: E/C from paired test trajectories
    rng = np.random.default_rng(args.seed)
    pt = {s: (np.asarray(aux[s]['per_trajectory']['E_aux_state']),
              np.asarray(aux[s]['per_trajectory']['C_state']))
          for s in SYSTEMS}
    boot_rho = {k: [] for k in keys + ['X_range_ratio']}
    boot_ec = {s: [] for s in SYSTEMS}
    for _ in range(args.boot):
        ecb = []
        for s in SYSTEMS:
            e, c = pt[s]
            sel = rng.integers(0, e.shape[1], e.shape[1])
            r = float(e[:, sel].mean() / c[:, sel].mean())
            ecb.append(r)
            boot_ec[s].append(r)
        ecb = np.array(ecb)
        for k in keys:
            v = np.array([per_sys[s]['psi'][k] for s in SYSTEMS])
            boot_rho[k].append(spearman(v, ecb))
        v = np.array([per_sys[s]['X_range_ratio'] for s in SYSTEMS])
        boot_rho['X_range_ratio'].append(spearman(v, ecb))

    boot = {k: {'frac_rho_ge_0p9': float(np.mean(np.array(v) >= 0.9)),
                'rho_median': float(np.median(v)),
                'rho_ci95': [float(np.percentile(v, 2.5)),
                             float(np.percentile(v, 97.5))]}
            for k, v in boot_rho.items()}
    ec_ci = {s: {'point': float(aux[s]['E_over_C']),
                 'ci95': [float(np.percentile(boot_ec[s], 2.5)),
                          float(np.percentile(boot_ec[s], 97.5))]}
             for s in SYSTEMS}

    # POST HOC, reported but NOT part of the verdict
    v = np.array([per_sys[s]['psi']['D3b_within_traj_resid_POSTHOC']
                  for s in SYSTEMS])
    rho, p, tot = exact_perm_p(v, ec)
    corr['D3b_within_traj_resid_POSTHOC'] = {
        'values': v.tolist(), 'rho': rho, 'exact_p': p, 'n_perm': tot,
        'passes_0p9': bool(rho >= 0.9),
        'role': ('POST HOC: devised after D3 failed, to separate within-run '
                 'folding from between-run spread. NOT counted in the verdict '
                 'and it cannot rescue the pre-registration.')}

    n_pass = sum(corr[k]['passes_0p9'] for k in keys[1:])   # D1, D2, D3
    n_pass_best = sum(corr[k]['rho_tiebreak_best'] >= 0.9 for k in keys[1:])
    verdict = ('PASS: all three alternative definitions order the systems '
               '(rho >= 0.9); the criterion is a property of the coordinate, '
               'not of the sign-counting statistic. P29b proceeds.'
               if n_pass == 3 else
               'PARTIAL: two of three alternative definitions order the '
               'systems; proceed to P29b with the failing definition named '
               'wherever the criterion is quoted.' if n_pass == 2 else
               'FAIL: the ordering does not survive redefinition. P29 stops '
               'here and is reported as a negative; P24 stays the descriptive '
               'split it already is.')

    out = {
        'systems': SYSTEMS,
        'E_over_C': ec.tolist(),
        'E_over_C_bootstrap': ec_ci,
        'per_system': per_sys,
        'correlations': corr,
        'bootstrap': {'n': args.boot, 'seed': args.seed, 'by_measure': boot},
        'n_alt_definitions_passing': int(n_pass),
        'n_alt_definitions_passing_best_tiebreak': int(n_pass_best),
        'verdict': verdict,
        'n5_caveat': ('exact two-sided permutation p over all 120 orderings; '
                      'p = 0.0167 is the FLOOR at n = 5 and a perfect ordering '
                      'cannot do better'),
        '_provenance': stamp('sc_injectivity.py',
                             inputs={'auxcoord.json': aux}),
    }
    path = os.path.join(OUT, 'injectivity.json')
    json.dump(out, open(path, 'w'), indent=1)

    print(f"{'system':9s} {'E/C':>7s}  " + '  '.join(f'{k[:14]:>14s}'
                                                     for k in keys))
    for i, s in enumerate(SYSTEMS):
        print(f'{s:9s} {ec[i]:7.2f}  ' + '  '.join(
            f"{per_sys[s]['psi'][k]:14.4f}" for k in keys))
    print()
    for k in keys + ['X_range_ratio']:
        c = corr[k]
        print(f"{k:22s} rho {c['rho']:+.3f}  exact p {c['exact_p']:.4f}  "
              f"boot P(rho>=0.9) {boot[k]['frac_rho_ge_0p9']:.3f}  "
              f"tie-break [{c['rho_tiebreak_worst']:+.3f}, "
              f"{c['rho_tiebreak_best']:+.3f}]")
    c = corr['D3b_within_traj_resid_POSTHOC']
    print(f"{'D3b (POST HOC)':22s} rho {c['rho']:+.3f}  "
          f"exact p {c['exact_p']:.4f}   -- not part of the verdict")
    print(f'\npassing alternative definitions: {n_pass} of 3 '
          f'({n_pass_best} of 3 under the most favourable tie-break)')
    print()
    print(verdict)
    print('wrote', path)


if __name__ == '__main__':
    main()
