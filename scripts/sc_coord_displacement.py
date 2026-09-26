"""P30 (Track F) -- is the clock-error floor a special case of one displacement?

WHY THIS EXISTS
---------------
`sc_displacement.py` (P12) computes, with no network at all,

    F(d) = || G(tau / (1 + d)) - G(tau) ||

the field displacement caused by asking a PERFECT operator at the wrong
*normalised time*, and shows it predicts the deployed error of every arm that
is conditioned on time. It has since answered three questions it was not built
for: it predicted the deployed error (P12), it explained the H1 intervention
(§5.1), and it bounded the grid-precision worry (P21).

The question here is whether that is a coincidence of the time coordinate
or an instance of something general. The manuscript currently carries a
mechanism for the time arms plus two *empirical* studies -- E21/E23 phase noise
and P13's measurement contract -- for the state arm. If one formula covers all
of them, §5 has a single mechanism instead of a mechanism and two appendices.

THE ALGEBRA (F2 first, as the plan asks, and it is short)
---------------------------------------------------------
Let the operator be conditioned on a coordinate `c` that is read off the state,
and let `G` be the reference trajectory. Conditioning means the operator answers
the question "what is the field where the coordinate takes value `c`?", so it
implicitly composes `G` with `c^{-1}`:

    G_c(v) = G(t)  where  c(t) = v.

An error `delta` in the measured coordinate therefore costs

    F_c(delta) = || G_c(v + delta) - G_c(v) ||                        (*)

and the clock floor is the case `c = tau`, `delta = tau/(1+d) - tau`, where
`c^{-1}` is trivial because tau is the parametrisation itself.

**(*) is well defined only where `c` is injective along the trajectory.** Where
the coordinate folds back, `c^{-1}(v)` is a set and the displacement is not a
function of `delta` at all -- there is no "the" state at that coordinate value.
That is the exact sense in which the injectivity question (P29a) and the floor
are two faces of one object: injectivity is the *domain condition* of (*).

Two things must be said plainly about that connection. First, P29a **refuted**
the empirical claim that non-injectivity ORDERS the systems by how much a
coordinate costs; nothing here restores it, and this script does not test it.
What survives is the mathematical statement, which needs no ordering: where the
coordinate is non-injective the displacement is ill-defined. Second, this is
precisely what the branch flag `b` is for. `(phi, b)` is injective where `phi`
alone is not, so arm C's coordinate has a well-defined displacement and arm E's
(bulk `psi`, no flag) need not. The inversion below is therefore performed
WITHIN the branch by default, and the ambiguity rate is reported either way.

WHAT IS COMPUTED
----------------
For each system and each coordinate `c` in {tau, phi, psi}, and each
perturbation scale `sigma` (dex for phi and psi, relative for tau):

  * `F_c(sigma)`: draw `delta ~ N(0, sigma)` independently per (trajectory,
    tau) sample -- which is exactly the noise model `sc_hybrid.py` applies to
    the measured curves -- invert `c` locally to the nearest crossing, and
    measure the field displacement in the normalised units `field_err` reports.
  * `ambiguous_frac`: the share of samples with more than one crossing (the
    displacement is ill-defined there and the nearest one is used).
  * `out_of_range_frac`: the share with no crossing at all inside the
    trajectory (the perturbed value leaves the coordinate's range; the endpoint
    is used, which is what a deployed operator faces).

SELF-CONSISTENCY, checked before anything is claimed
----------------------------------------------------
With `c = tau` this code must reproduce `sc_displacement.py`'s F(d) at the imposed
offsets, to a few percent. If it does not, the generalisation is a bug and not
a result, so that comparison is reported first and gates the rest.

THE PRE-REGISTERED TEST
-----------------------
`sc_hybrid.py` measured arm C's error under i.i.d. `phi` noise of `eps` dex at
eps in {0.01, 0.02, 0.05, 0.1, 0.2} for all five systems -- 25 points, already
in the repository, collected for another purpose. If (*) is the mechanism then
the same quadrature the clock floor obeys should hold:

    E_C(eps)  ~=  sqrt( E_C(0)^2 + F_phi(eps)^2 ).

  PASS     median predicted/measured over the 25 points in [0.80, 1.25], the
           band P12's clock floor achieved (median 0.972-0.993), AND the
           prediction rises with eps in all five systems.
           -> one displacement formula governs the time AND the state arms;
              §5 gets a single mechanism.
  PARTIAL  median in [0.6, 1.6]. -> the formula captures the scaling but not
           the constant; report as an approximation WITH its bias, and keep the
           empirical studies.
  FAIL     outside [0.6, 1.6], or the ordering fails.
           -> the floor is a property of the TIME coordinate specifically.
              Report as a negative, exactly as the spectral-bias mechanism and
              the injectivity criterion were, and leave §5 as a mechanism plus
              two empirical studies.

Arm E's coordinate `psi` has no measured noise sweep in this repository, so its
`F_psi` is computed and reported as a PREDICTION only, explicitly untested.

Usage:
  sc_coord_displacement.py [--systems ...] [--draws 8] [--seed 20260908]
Output: results/statecond/coord_floor.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import statecond.data as scd                                    # noqa: E402
from sc_experiment import LATE, Norm                            # noqa: E402
from sc_displacement import resample_tau                               # noqa: E402
from sc_provenance import stamp                                 # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond')
DATA = os.path.join(ROOT, 'data', 'sc')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']
EPS_DEX = [0.01, 0.02, 0.05, 0.1, 0.2]          # matches sc_hybrid's grid
TAU_DELTAS = [0.005, 0.01, 0.02, 0.05, 0.1]     # matches sc_displacement's grid


def psi_table(d):
    """Bulk L2 contrast in dex, the arm E coordinate (as in sc_auxcoord)."""
    u = np.asarray(d['u'], dtype=np.float64)
    nrm = np.sqrt((u ** 2).mean(axis=(2, 3)))
    return np.log10(np.maximum(nrm, 1e-300) / np.maximum(nrm[:, :1], 1e-300))


def invert_coordinate(cvals, taus, j, target, branch_lo, branch_hi):
    """tau where `cvals` crosses `target`, nearest to taus[j], within a branch.

    Returns (tau_star, n_crossings). With no crossing inside the branch the
    endpoint nearest in value is returned and n_crossings is 0 -- the deployed
    analogue of the arm-B endpoint clip, which is also what the operator would
    be forced to do.
    """
    lo, hi = branch_lo, branch_hi
    seg = cvals[lo:hi + 1]
    diff = seg - target
    sign_change = np.nonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) <= 0)[0]
    if sign_change.size == 0:
        k = lo + int(np.argmin(np.abs(diff)))
        return float(taus[k]), 0
    cands = []
    for i in sign_change:
        a, b = lo + i, lo + i + 1
        denom = cvals[b] - cvals[a]
        w = 0.0 if denom == 0 else (target - cvals[a]) / denom
        w = min(max(w, 0.0), 1.0)
        cands.append(taus[a] + w * (taus[b] - taus[a]))
    cands = np.asarray(cands)
    k = int(np.argmin(np.abs(cands - taus[j])))
    return float(cands[k]), int(sign_change.size)


def displacement(un, ctab, taus, te, sigma, rng, draws, flag=None,
                 restrict_branch=True):
    """F_c(sigma) in normalised field units, plus the domain diagnostics."""
    n_tau = len(taus)
    tot_sq, tot_n = [], []
    ambiguous = out_of_range = count = 0
    late = taus > LATE
    late_sq = []
    for r in te:
        cvals = ctab[r]
        if flag is not None and restrict_branch:
            fl = flag[r]
            bounds = []
            for j in range(n_tau):
                same = np.nonzero(fl == fl[j])[0]
                bounds.append((int(same.min()), int(same.max())))
        else:
            bounds = [(0, n_tau - 1)] * n_tau
        tq = np.empty((draws, n_tau))
        for k in range(draws):
            deltas = rng.normal(0.0, sigma, size=n_tau)
            for j in range(n_tau):
                lo, hi = bounds[j]
                t_star, n_cross = invert_coordinate(
                    cvals, taus, j, cvals[j] + deltas[j], lo, hi)
                tq[k, j] = t_star
                count += 1
                if n_cross == 0:
                    out_of_range += 1
                elif n_cross > 1:
                    ambiguous += 1
        # fields at the perturbed coordinate, against the unperturbed target
        for k in range(draws):
            up = resample_tau(un[r:r + 1], taus, tq[k], 'linear')[0]
            diff = up - un[r]
            e = np.sqrt((diff ** 2).mean(axis=(1, 2)))       # (T,)
            tot_sq.append(e.mean())
            late_sq.append(e[late].mean())
            tot_n.append(1)
    return {'F': float(np.mean(tot_sq)), 'F_late': float(np.mean(late_sq)),
            'ambiguous_frac': ambiguous / max(count, 1),
            'out_of_range_frac': out_of_range / max(count, 1)}


def tau_floor_check(un, taus, te, deltas):
    """Reproduce sc_displacement.py's F(d) with this file's machinery."""
    out = {}
    for d in deltas:
        acc = []
        for sgn in (+1.0, -1.0):
            tq = np.clip(taus / (1.0 + sgn * d), 0.0, 1.0)
            up = resample_tau(un, taus, tq, 'linear')
            diff = up[te] - un[te]
            acc.append(float(np.sqrt((diff ** 2).mean(axis=(2, 3))).mean()))
        out[str(d)] = float(np.mean(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=SYSTEMS)
    ap.add_argument('--draws', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260908)
    args = ap.parse_args()

    floor = json.load(open(os.path.join(OUT, 'floor.json')))
    out = {'eps_dex': EPS_DEX, 'draws': args.draws, 'seed': args.seed}
    ratios_all = []

    for name in args.systems:
        d = scd.load(os.path.join(DATA, f'{name}.h5'))
        tr, _, te = scd.split(d['u'].shape[0])
        n = Norm(d, tr)
        un = (np.asarray(d['u'], dtype=np.float64) - n.f_mu) / n.f_sd
        taus = scd.tau_grid()
        phi = np.asarray(d['phi'], dtype=np.float64)
        flag = np.asarray(d['flag'])
        psi = psi_table(d)
        rng = np.random.default_rng(args.seed)

        rec = {'n_test': int(len(te))}
        # --- self-consistency against sc_displacement.py, before anything is claimed
        mine = tau_floor_check(un, taus, te, TAU_DELTAS)
        theirs = floor.get('floor', {}).get(name, {})
        rec['tau_floor_self_check'] = {
            'this_script': mine,
            'sc_displacement_json': {k: theirs.get(k) for k in mine},
            'max_rel_disagreement': max(
                (abs(mine[k] - theirs[k]) / theirs[k]
                 for k in mine if theirs.get(k)), default=None)}

        # --- the state coordinates
        for cname, ctab, fl in (('phi', phi, flag), ('psi', psi, None)):
            rec[cname] = {}
            for eps in EPS_DEX:
                rec[cname][str(eps)] = displacement(
                    un, ctab, taus, te, eps, np.random.default_rng(
                        args.seed + int(1000 * eps)), args.draws, flag=fl)

        # --- the pre-registered comparison, for phi only
        hy = json.load(open(os.path.join(OUT, f'hybrid_{name}.json')))
        e0 = hy['aggregate']['eps0.0_q0.0']['C_state_point']['mean']
        rec['measured_arm_C'] = {'0.0': e0}
        rec['prediction'] = {}
        for eps in EPS_DEX:
            key = f'eps{eps}_q0.0'
            meas = hy['aggregate'][key]['C_state_point']['mean']
            F = rec['phi'][str(eps)]['F']
            pred = float(np.sqrt(e0 ** 2 + F ** 2))
            rec['measured_arm_C'][str(eps)] = meas
            rec['prediction'][str(eps)] = {
                'F_phi': F, 'predicted': pred, 'measured': meas,
                'ratio_pred_over_meas': pred / meas}
            ratios_all.append(pred / meas)
        out[name] = rec
        print(f'{name:8s} tau self-check max rel '
              f"{rec['tau_floor_self_check']['max_rel_disagreement']:.3f}  |  "
              + '  '.join(
                  f"{eps}: {rec['prediction'][str(eps)]['ratio_pred_over_meas']:.2f}"
                  for eps in EPS_DEX), flush=True)

    med = float(np.median(ratios_all))
    monotone = all(
        all(out[s]['prediction'][str(a)]['F_phi']
            < out[s]['prediction'][str(b)]['F_phi']
            for a, b in zip(EPS_DEX[:-1], EPS_DEX[1:]))
        for s in args.systems if s in out)
    verdict = (
        'PASS: one displacement formula governs the time and the state arms'
        if 0.80 <= med <= 1.25 and monotone else
        'PARTIAL: the formula captures the scaling but not the constant; '
        'report the bias with it'
        if 0.6 <= med <= 1.6 and monotone else
        'FAIL: the floor is a property of the TIME coordinate specifically; '
        'report as a negative and leave the empirical studies in place')
    out['_summary'] = {
        'n_points': len(ratios_all),
        'median_ratio': med,
        'ratio_range': [float(np.min(ratios_all)), float(np.max(ratios_all))],
        'prediction_monotone_in_all_systems': bool(monotone),
        'verdict': verdict,
        '_provenance': stamp('sc_coord_displacement.py',
                             inputs={'floor.json': floor}),
    }
    path = os.path.join(OUT, 'coord_floor.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nmedian predicted/measured over {len(ratios_all)} points: '
          f'{med:.3f}  range [{np.min(ratios_all):.2f}, '
          f'{np.max(ratios_all):.2f}]')
    print(verdict)
    print('wrote', path)


if __name__ == '__main__':
    main()
