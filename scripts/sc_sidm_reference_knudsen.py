"""P21b: grid convergence of the sidm reference t_sing for KNUDSEN-terminated runs.

P21 (`sc_sidm_reference.py`) converged only runs that end on the density criterion:
87 of its 112 prior draws ended on the Knudsen floor or the horizon and were skipped, so
the published sidm row covers about a fifth of the training prior. For status-2 runs the
benchmark label is t(min_i Kn_i = kn_min) (`gravop.dataset.sample_from_run`). Kn is a
local, grid-intensive quantity, so the solver's own stopping time is a grid-independent
definition and can be compared across grids directly (no fixed-shell anchoring needed).

Protocol (frozen before running): fresh prior draws from `draw_params(rng, 'train')`,
seed 20261105, classify the production run (300 shells) by status; for the first 24
status-2 draws, rerun at 600 and 1200 shells and at the production grid with eps_t
tightened 3x (3e-4 -> 1e-4, P21's choice). Report the status mix of all attempted draws,
production-vs-finest relative change of t_sing, a Richardson estimate where the three grids
give a monotone sequence, the field disagreement in dex, and the margin against the
converged sidm clock head (0.0423%). Nothing is fitted; no benchmark label is changed.

Output: results/statecond/sidm_reference_knudsen.json
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.dirname(__file__))
from gen_dataset import SOLVER_KW, draw_params, make_ic  # noqa: E402
from sc_sidm_reference import field_dex  # noqa: E402
from gravop import solver  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'statecond', 'sidm_reference_knudsen.json')
HEAD_ERR = 0.0423431396484375 / 100  # converged sidm clock head, sidm_reference.json


def run(args):
    d, ic, rc, n_shells, eps_t = args
    kw = dict(SOLVER_KW); kw['n_shells'] = n_shells; kw['eps_t'] = eps_t
    prof = make_ic(ic, rc, n_shells)
    if prof is not None: kw['r'], kw['m'], kw['rho'], kw['p'] = prof
    t0 = time.time(); res = solver.run_halo(**d, **kw)
    out = {k: res[k] for k in ['t', 'r', 'rho']}; out.update(status=int(res['status']), t_final=float(res['t_final']), wall=time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=24); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20261105); ap.add_argument('--max-draws', type=int, default=200); a = ap.parse_args()
    rng = np.random.default_rng(a.seed); draws = [draw_params(rng, 'train') for _ in range(a.max_draws)]
    t0 = time.time(); status_mix = {}; selected = []
    with ProcessPoolExecutor(a.workers) as ex:
        # production runs in batches until n Knudsen-terminated draws are found
        i = 0
        while len(selected) < a.n and i < len(draws):
            batch = list(range(i, min(i + a.workers * 2, len(draws)))); i = batch[-1] + 1
            for j, r in zip(batch, ex.map(run, [(*draws[j], 300, SOLVER_KW['eps_t']) for j in batch])):
                key = 'unusable' if r['status'] in (3, 4) or r['t_final'] < 1.0 else {0: 'density', 1: 'horizon', 2: 'knudsen'}[r['status']]
                status_mix[key] = status_mix.get(key, 0) + 1
                if key == 'knudsen' and len(selected) < a.n: selected.append((j, r))
        attempted = i
        jobs = [(*draws[j], g, SOLVER_KW['eps_t']) for j, _ in selected for g in (600, 1200)] + [(*draws[j], 300, 1e-4) for j, _ in selected]
        res = list(ex.map(run, jobs))
    n = len(selected); fine = {600: res[0:2 * n:2], 1200: res[1:2 * n:2]}; tol = res[2 * n:]
    rows = []
    for k, (j, rp) in enumerate(selected):
        r6, r12, rt = fine[600][k], fine[1200][k], tol[k]
        t = {300: rp['t_final'], 600: r6['t_final'], 1200: r12['t_final']}
        row = dict(draw=j, params={kk: (float(v) if isinstance(v, (int, float)) else v) for kk, v in draws[j][0].items()}, ic=draws[j][1],
                   rc=float(draws[j][2] or 0.), t_final=t, status={300: rp['status'], 600: r6['status'], 1200: r12['status'], 'tol': rt['status']},
                   rel_dt_600=abs(t[300] - t[600]) / t[600], rel_dt_1200=abs(t[300] - t[1200]) / t[1200],
                   rel_dt_tol=abs(t[300] - rt['t_final']) / rt['t_final'])
        c, m, f = t[300], t[600], t[1200]
        if (m - f) != 0 and (c - m) / (m - f) > 0:
            p = float(np.log((c - m) / (m - f)) / np.log(2.)); ext = f + (f - m) / (2 ** p - 1)
            row.update(observed_order=p, t_richardson=float(ext), rel_dt_richardson=abs(c - ext) / abs(ext))
        for g, rr in [(600, r6), (1200, r12)]:
            A = dict(rp, t_c=rp['t_final']); B = dict(rr, t_c=rr['t_final']); row[f'field_dex_{g}'] = field_dex(A, B)
        rows.append(row)
    same_status = all(all(v == 2 for v in r['status'].values()) for r in rows)
    rel = np.array([r['rel_dt_1200'] for r in rows]); rich = np.array([r['rel_dt_richardson'] for r in rows if 'rel_dt_richardson' in r])
    out = dict(protocol=__doc__.split('Output:')[0].strip(), seed=a.seed, attempted_draws=attempted, status_mix_production=status_mix,
               n_trajectories=n, all_grids_end_on_knudsen=bool(same_status),
               rel_dt_sing_median_vs1200=float(np.median(rel)), rel_dt_sing_max_vs1200=float(rel.max()),
               rel_dt_sing_median_vs600=float(np.median([r['rel_dt_600'] for r in rows])),
               richardson_rows=int(len(rich)), rel_dt_richardson_median=float(np.median(rich)) if len(rich) else None,
               observed_order_median=float(np.median([r['observed_order'] for r in rows if 'observed_order' in r])) if len(rich) else None,
               tolerance_rel_median=float(np.median([r['rel_dt_tol'] for r in rows])),
               field_dex_median_vs600=float(np.median([r['field_dex_600'] for r in rows])),
               converged_head_err=HEAD_ERR, margin_median_vs1200=float(HEAD_ERR / np.median(rel)),
               margin_median_richardson=float(HEAD_ERR / np.median(rich)) if len(rich) else None,
               wall_seconds=time.time() - t0, rows=rows)
    tmp = OUT + '.tmp'; json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, OUT)
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'protocol')}, indent=1))


if __name__ == '__main__': main()
