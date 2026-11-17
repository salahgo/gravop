"""Does the gentler conductivity interpolation (alpha = 0.5) fit our N-body run better?

van den Bosch & Tokayer (2026, arXiv:2610.28890) calibrate the gravothermal fluid model
against a spherical DSMC code and report that the harmonic interpolation between short-
and long-mean-free-path conductivities (alpha = 1, used by gravop.solver) is inaccurate;
K = (K_s^-alpha + K_l^-alpha)^(-1/alpha) with alpha = 0.5 fits with a single LMFP
parameter. Here the same question is asked of the project's own isolated-halo N-body
calibration (results/e2, Fischer, Yu & Dolag 2025, N = 5e7).

Protocol (fixed before running): for alpha in {1, 0.5}, scan C over 0.50..1.20 step 0.01,
run the production configuration of scripts/e2_calibration_fit.py at each C with NO time
rescaling, and score the RMS log-density residual over the frozen calibration snapshots
010-045 (E2's scorer, unchanged). Report the best C and residual per alpha, and the
residual on snapshot 051 (outside the fitted set) at each alpha's best C when the run
reaches it. alpha = 1 must reproduce the stored production run bit for bit and the stored
direct residual at C = 0.712. One N-body halo; N-body softening may bias it (the cited paper
finds N-body converges only for small softening). Nothing in src/, results/ or paper/ is
changed. Output: results/e2/conductivity_alpha/alpha_<alpha>.json, collected by
scripts/e2_conductivity_interpolation.py

Usage: SOLVER_ALPHA=1.0 python run_alpha.py ; SOLVER_ALPHA=0.5 python run_alpha.py
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'results', 'e2', 'conductivity_alpha')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts')); sys.path.insert(0, os.path.join(ROOT, 'src'))
import solver_alpha as solver  # noqa: E402  (reads SOLVER_ALPHA at import)
import e2_calibration_fit as e2  # noqa: E402  (scorer and frozen run settings only)

with np.load(os.path.join(ROOT, 'results', 'e2', 'nbody_profiles.npz')) as _z:
    D = {k: _z[k] for k in _z.files}  # in memory: a lazy NpzFile must not be shared across forked workers


def score_snapshot(run, name):
    keep = e2.CALIBRATION_SNAPSHOTS
    try:
        e2.CALIBRATION_SNAPSHOTS = (name,); p = e2.score(run, D, 1.0)
    finally:
        e2.CALIBRATION_SNAPSHOTS = keep
    return None if p is None else p[0]


def one(C):
    t0 = time.time(); run = solver.run_halo(C=float(C), **e2.RUN_KW); p = e2.score(run, D, 1.0)
    return dict(C=float(C), t_final=float(run['t_final']), status=int(run['status']), rms_dex=None if p is None else e2.total(p),
                per_snapshot=None if p is None else dict(zip(e2.CALIBRATION_SNAPSHOTS, p)),
                snapshot_051=score_snapshot(run, 'snapshot_051'), wall=time.time() - t0)


def main():
    alpha = solver.ALPHA; out = {'protocol': __doc__.split('Usage:')[0].strip(), 'alpha': alpha}
    if alpha == 1.0:
        run = solver.run_halo(C=e2.C_PROD, **e2.RUN_KW); st = np.load(os.path.join(ROOT, 'results', 'e2', 'fluid_run_production.npz'))
        out['reproduces_stored_production_run'] = bool(run['t'].shape == st['t'].shape and all(np.array_equal(run[k], st[k]) for k in ('t', 'r', 'rho', 'p')))
        if not out['reproduces_stored_production_run']: raise SystemExit('exploratory copy does not reproduce the production run')
    grid = np.round(np.arange(.50, 1.2001, .01), 2)
    with ProcessPoolExecutor(int(os.environ.get('WORKERS', '6'))) as ex: rows = list(ex.map(one, grid))
    ok = [r for r in rows if r['rms_dex'] is not None]; coarse = min(ok, key=lambda r: r['rms_dex'])
    # refinement (added after a first run whose grid lacked the stored C = 0.712; protocol otherwise unchanged)
    fine_grid = [c for c in np.round(np.arange(coarse['C'] - .02, coarse['C'] + .0201, .002), 3) if abs(c - round(c, 2)) > 1e-9]
    ref = json.load(open(os.path.join(ROOT, 'results', 'e2', 'calibration_fit.json')))['direct_conductivity']['C_eff_coarse']
    extra = sorted(set(fine_grid) | ({ref['C']} if alpha == 1.0 else set()))
    with ProcessPoolExecutor(int(os.environ.get('WORKERS', '6'))) as ex: rows += list(ex.map(one, extra))
    ok = [r for r in rows if r['rms_dex'] is not None]; best = min(ok, key=lambda r: r['rms_dex'])
    out.update(scan=rows, coarse_best=coarse, best=best)
    if alpha == 1.0:
        mine = [r for r in rows if abs(r['C'] - ref['C']) < 1e-9][0]
        out['reproduces_stored_direct_residual'] = bool(abs(mine['rms_dex'] - ref['rms_dex_no_rescaling']) < 1e-12)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f'alpha_{alpha}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('scan', 'protocol')}, indent=1))


if __name__ == '__main__': main()
