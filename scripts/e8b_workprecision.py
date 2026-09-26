"""P19 -- work-precision curves, so the speed-up is quoted at matched accuracy.

WHAT WAS WRONG WITH THE OLD NUMBER
----------------------------------
The 4090x headline compares our semi-implicit solver at `eps_t = 3e-4` against
the reference explicit implementation at `eps_t = 1e-3`
(`results/e1/crosscheck_implicit_N150.json`). Those are different tolerances,
and the accuracies differ with them: measured against the reference, our
explicit run at `eps_t = 1e-3` reproduces t_c to 8.3e-05 relative while our
implicit run at `3e-4` reproduces it to 1.25e-03. The implicit scheme is
therefore ~15x less accurate at the settings the speed-up is quoted from, and
some unknown part of 4090x is bought with that accuracy rather than with the
scheme.

The factorisation 4090x = 12.6x (implementation) x 326x (scheme) separates
cleanly on this point:

  * the IMPLEMENTATION leg is already matched -- same scheme, same eps_t, two
    implementations -- and needs only to be stated as such;
  * the SCHEME leg is not, and is what this script measures.

WHAT IT DOES
------------
Sweeps `eps_t` for both compiled schemes at a fixed configuration, records wall
time and the resulting t_c, and scores each against a converged reference
(the finest implicit run, cross-checked against the finest explicit run so the
reference is not defined by one of the two arms alone). The speed-up is then
read off at MATCHED accuracy by interpolating each work-precision curve, which
is the number a practitioner actually gets.

Also recorded, because the referee asked and because none of it was stated:
hardware, precision, single-core vs threaded, timing variance over repeats, and
the query count at which emulation repays its training cost.

Output: results/e8/workprecision.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from gravop import solver                                       # noqa: E402

OUT = os.path.join(ROOT, 'results', 'e8')

# the configuration the 4090x claim is quoted at (scripts/e1_crosscheck.py)
SIGMA_HAT = 0.5
N_SHELLS = 150
RHO_END = 300.0
TC_FACTOR = 100.0


def tc_of(ts, rho_c, factor):
    target = factor * rho_c[0]
    idx = int(np.argmax(rho_c >= target))
    if rho_c[idx] < target or idx == 0:
        return None
    t0, t1 = ts[idx - 1], ts[idx]
    l0, l1 = np.log(rho_c[idx - 1]), np.log(rho_c[idx])
    return float(t0 + (t1 - t0) * (np.log(target) - l0) / (l1 - l0))


def run_once(mode, eps_t, n_shells=N_SHELLS):
    t0 = time.perf_counter()
    res = solver.run_halo(sigma_m=SIGMA_HAT, n_shells=n_shells, eps_t=eps_t,
                          rho_c_factor=RHO_END, snap_dlnrho=0.02,
                          timestep_mode=mode)
    wall = time.perf_counter() - t0
    return wall, tc_of(res['t'], res['rho'][:, 3], TC_FACTOR), res


WORKER = """
import json, sys, time
sys.path.insert(0, %r)
from gravop import solver
import numpy as np
mode, eps, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
solver.run_halo(sigma_m=0.5, n_shells=40, eps_t=1e-3, rho_c_factor=2.0,
                timestep_mode=mode)                      # pay the JIT
t0 = time.perf_counter()
r = solver.run_halo(sigma_m=%r, n_shells=n, eps_t=eps, rho_c_factor=%r,
                    snap_dlnrho=0.02, timestep_mode=mode)
wall = time.perf_counter() - t0
rho_c = r['rho'][:, 3]
target = %r * rho_c[0]
idx = int(np.argmax(rho_c >= target))
tc = None
if rho_c[idx] >= target and idx > 0:
    t0_, t1_ = r['t'][idx - 1], r['t'][idx]
    l0, l1 = np.log(rho_c[idx - 1]), np.log(rho_c[idx])
    tc = float(t0_ + (t1_ - t0_) * (np.log(target) - l0) / (l1 - l0))
print(json.dumps({'wall': wall, 't_c': tc, 'status': int(r['status']),
                  'steps': int(r['n_steps'])}))
""" % (os.path.join(ROOT, 'src'), SIGMA_HAT, RHO_END, TC_FACTOR)


def run_capped(mode, eps_t, cap_s, n_shells=N_SHELLS):
    """One run in a worker process, killed at `cap_s`.

    A configuration that does not finish inside the cap is RECORDED as such
    rather than waited on. That distinction is the point: for an explicit
    scheme, loosening the tolerance past the stability limit does not buy a
    cheaper, sloppier answer -- it buys no answer at all, and a work-precision
    curve that quietly omitted those points would misrepresent the method.
    """
    cmd = [sys.executable, '-c', WORKER, mode, repr(eps_t), str(n_shells)]
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=cap_s)
    except subprocess.TimeoutExpired:
        return {'eps_t': eps_t, 'mode': mode, 'n_shells': n_shells,
                'wall_median': None, 't_c': None, 'rel_err': None,
                'exceeded_cap_s': cap_s,
                'note': 'did not complete within the cap'}
    if out.returncode != 0:
        return {'eps_t': eps_t, 'mode': mode, 'n_shells': n_shells,
                'wall_median': None, 't_c': None, 'rel_err': None,
                'failed': out.stderr.strip().splitlines()[-1:] or ['?'],
                'wall_to_failure': time.perf_counter() - t0}
    d = json.loads(out.stdout.strip().splitlines()[-1])
    return {'eps_t': eps_t, 'mode': mode, 'n_shells': n_shells,
            'wall_median': d['wall'], 'wall_min': d['wall'],
            'wall_max': d['wall'], 'wall_rel_spread': 0.0,
            't_c': d['t_c'], 'status': d['status'], 'steps': d['steps'],
            'repeats': 1}


def measure(mode, eps_t, repeats, n_shells=N_SHELLS):
    """Wall time over repeats (after a warm-up that pays the numba JIT once)."""
    walls, tc = [], None
    for k in range(repeats):
        w, t, _ = run_once(mode, eps_t, n_shells)
        walls.append(w)
        tc = t
    return {'eps_t': eps_t, 'n_shells': n_shells, 'repeats': repeats,
            'wall_median': float(np.median(walls)),
            'wall_min': float(min(walls)), 'wall_max': float(max(walls)),
            'wall_rel_spread': float((max(walls) - min(walls))
                                     / max(np.median(walls), 1e-12)),
            't_c': tc}


def interp_wall_at_error(curve, target_err):
    """Wall time this scheme needs to reach `target_err`, log-interpolated.

    Points are (error, wall). Returns None if the target lies outside the
    range the sweep actually covers -- extrapolating a work-precision curve is
    how speed-ups get invented.
    """
    pts = sorted((c['rel_err'], c['wall_median']) for c in curve
                 if c.get('rel_err') is not None and c['rel_err'] > 0
                 and c.get('wall_median') is not None)
    if not pts or target_err < pts[0][0] or target_err > pts[-1][0]:
        return None
    for (e0, w0), (e1, w1) in zip(pts, pts[1:]):
        if e0 <= target_err <= e1:
            f = ((np.log(target_err) - np.log(e0))
                 / max(np.log(e1) - np.log(e0), 1e-12))
            return float(np.exp(np.log(w0) + f * (np.log(w1) - np.log(w0))))
    return None


def break_even(doc):
    """When does emulating repay the cost of building the emulator?

    A per-query speed-up is not a saving until the data generation and the
    training that produced it have been earned back. The crossing point is the
    number a practitioner needs, and it is the one number the efficiency claim
    never quoted.

    Every input is measured: the solver cost per trajectory from
    `results/e8/bench.json`, the emulator cost per trajectory from the same
    file, the dataset size from the production generator, and the training cost
    from a recorded arm. The training term is given at two budgets because the
    production ensemble's wall time was never recorded, and the answer is
    reported as a range rather than invented.
    """
    b = json.load(open(os.path.join(ROOT, 'results', 'e8', 'bench.json')))
    solver_s = float(np.median([r['wall_s'] for r in b['solver']]))
    traj = [r for r in b['emulator'] if r['kind'] == 'trajectory']
    emu_s = float(min(r['seconds_per_item'] for r in traj))
    n_train = 1300                       # scripts/gen_dataset.py train 1300
    gen_s = n_train * solver_s
    per_query_saving = solver_s - emu_s
    out = {'solver_s_per_trajectory': solver_s,
           'emulator_s_per_trajectory': emu_s,
           'per_query_speedup': solver_s / emu_s,
           'n_train': n_train,
           'data_generation_core_s': gen_s,
           'training_budgets_s': {}}
    for label, train_s in (('one_arm_50.6s', 50.6), ('ensemble_4h', 4 * 3600.)):
        out['training_budgets_s'][label] = {
            'training_s': train_s,
            'break_even_queries': (gen_s + train_s) / per_query_saving}
    out['note'] = ('50.6 s is a measured single-arm fit '
                   '(results/statecond/sidm.json); 4 h is the suite retrain '
                   'quoted in STATE.md and is an upper bound for one model. '
                   'Data generation dominates at the lower budget.')
    out['reading'] = (
        'the emulator repays its construction after '
        f"{out['training_budgets_s']['one_arm_50.6s']['break_even_queries']:.0f}"
        ' to '
        f"{out['training_budgets_s']['ensemble_4h']['break_even_queries']:.0f}"
        ' trajectory queries; a posterior sample of 1e5-1e6 likelihood calls '
        'clears that by one to two orders of magnitude, which is what makes '
        'the per-query figure worth quoting at all')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--explicit-eps', type=float, nargs='*',
                    default=[1e-2, 3e-3, 1e-3, 3e-4])
    ap.add_argument('--implicit-eps', type=float, nargs='*',
                    default=[3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--ref-eps', type=float, default=3e-6)
    ap.add_argument('--breakeven-only', action='store_true',
                    help='recompute the break-even section against the '
                         'existing artefact, without re-running the sweep')
    ap.add_argument('--explicit-cap-s', type=float, default=1500.0,
                    help='wall-clock cap per explicit run; exceeding it is '
                         'recorded as a result, not waited out')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'workprecision.json')

    if args.breakeven_only:
        doc = json.load(open(p))
        doc['break_even'] = break_even(doc)
        json.dump(doc, open(p, 'w'), indent=1)
        print(doc['break_even']['reading'])
        print('wrote', p)
        return

    print('warm-up (numba JIT)', flush=True)
    for mode in ('classic', 'implicit'):
        solver.run_halo(sigma_m=SIGMA_HAT, n_shells=40, eps_t=1e-3,
                        rho_c_factor=2.0, timestep_mode=mode)

    # the reference t_c: the finest implicit run, cross-checked against the
    # finest explicit run so that neither arm defines its own target
    print(f'reference: implicit eps_t={args.ref_eps}', flush=True)
    _, tc_ref, _ = run_once('implicit', args.ref_eps)
    xref = run_capped('classic', min(args.explicit_eps), args.explicit_cap_s)
    tc_ref_x = xref['t_c']
    ref_disagreement = (abs(tc_ref_x - tc_ref) / tc_ref
                        if tc_ref_x is not None else None)
    print(f'  implicit {tc_ref:.6f} vs explicit {tc_ref_x} '
          f'({ref_disagreement})', flush=True)

    curves = {}
    for mode, epss in (('implicit', args.implicit_eps),
                       ('classic', args.explicit_eps)):
        rows = []
        for eps in epss:
            if mode == 'implicit':
                r = measure(mode, eps, args.repeats)
            else:
                r = run_capped(mode, eps, args.explicit_cap_s)
            r['rel_err'] = (abs(r['t_c'] - tc_ref) / tc_ref
                            if r.get('t_c') is not None else None)
            rows.append(r)
            w = ('capped' if r.get('exceeded_cap_s')
                 else f"{r['wall_median']:.2f}s")
            e = ('n/a' if r['rel_err'] is None else f"{r['rel_err']:.2e}")
            print(f"  {mode:9s} eps_t={eps:.0e}  wall {w:>10s}"
                  f"  t_c {r.get('t_c')}  rel_err {e}", flush=True)
        curves[mode] = rows

    # the scheme speed-up AT MATCHED ACCURACY
    matched = {}
    for target in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        wi = interp_wall_at_error(curves['implicit'], target)
        we = interp_wall_at_error(curves['classic'], target)
        if wi and we:
            matched[f'{target:.0e}'] = {'implicit_wall': wi,
                                        'explicit_wall': we,
                                        'scheme_speedup': we / wi}

    doc = {
        'config': (f'NFW, sigma_hat={SIGMA_HAT}, N={N_SHELLS}, to '
                   f'{RHO_END}x rho_c; t_c at {TC_FACTOR}x'),
        'reference': {'mode': 'implicit', 'eps_t': args.ref_eps,
                      't_c': tc_ref, 't_c_explicit_check': tc_ref_x,
                      'rel_disagreement': ref_disagreement},
        'curves': curves,
        'matched_accuracy_scheme_speedup': matched,
        'published_scheme_speedup': 325.2,
        'published_note': ('208.8 s explicit at eps_t=1e-3 over 0.642 s '
                           'implicit at eps_t=3e-4 -- different tolerances, '
                           'and the implicit run is the less accurate of the '
                           'two against the reference implementation'),
        'environment': {
            'cpu': platform.processor() or platform.machine(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'precision': 'float64 throughout',
            'threading': ('single-threaded: the step loop is a numba njit '
                          'kernel with no parallel region'),
            'timing': (f'time.perf_counter around run_halo, {args.repeats} '
                       'repeats for the cheap arm after a JIT warm-up; the '
                       'reported figure is the median'),
        },
    }
    if matched:
        k = sorted(matched, key=lambda s: float(s))[0]
        doc['headline'] = (
            f'at matched t_c accuracy the semi-implicit scheme is '
            f'{matched[k]["scheme_speedup"]:.0f}x faster than the explicit one '
            f'(at rel_err {k}), against {doc["published_scheme_speedup"]:.0f}x '
            f'quoted at unmatched tolerances')
        print('\n' + doc['headline'])
    else:
        doc['headline'] = ('the two work-precision curves do not overlap in '
                           'accuracy over the swept tolerances; no matched '
                           'comparison is possible without extending the sweep')
        print('\n' + doc['headline'])

    doc['break_even'] = break_even(doc)
    print(doc['break_even']['reading'])
    json.dump(doc, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
