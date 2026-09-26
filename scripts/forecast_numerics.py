"""F5 targeted solver diagnostics, with F4 identities frozen before new solves."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time

import numpy as np

from forecast_integrate import Bank, DyadicMesh, log_likelihood
from forecast_calibrate import ROOT, TIMES, NOISE_DEX, R_OBS, SOLVER_KW, solver, ds, write_npz
from forecast_validate import digest, write_json

DATA = ROOT / 'data/d1/forecast_numerics'
OUT = ROOT / 'results/forecast_numerics'
F4 = ROOT / 'data/d1/forecast_integration'
LEVELS = {1: dict(eps_t=1e-4, snap_dlnrho=.01, snap_dlnt=.025),
          2: dict(eps_t=3e-5, snap_dlnrho=.005, snap_dlnt=.0125)}


def selected_cells():
    bank = Bank()
    obs = np.load(F4 / 'observations.npz')
    mesh = DyadicMesh(cells=json.loads((F4 / 'mesh.json').read_text())['cells'])
    keys, indices = mesh.stencil()
    ids = bank.ids(keys)
    lp = log_likelihood(bank.d['profiles'], bank.d['valid'], obs)
    values = np.exp(lp[ids] - lp.max(0))
    assessment = mesh.assess(values, indices)
    fine = sum(w * values[indices[:, k]] for k, w in
               enumerate(np.outer([1, 2, 1], [1, 2, 1]).ravel() / 16))
    coarse = sum(.25 * values[indices[:, k]] for k in (0, 2, 6, 8))
    error = np.abs(fine - coarse) * (mesh.cells[:, 2] / mesh.lattice)[:, None] ** 2
    error /= assessment['integral']
    cells = []
    for mask in (mesh.cells[:, 2] == 2, mesh.cells[:, 2] >= 4):
        eligible = np.flatnonzero(mask)
        cells.extend(eligible[np.argsort(assessment['priority'][eligible])[-3:]].tolist())
    for j in range(3):
        cases = np.flatnonzero(obs['time_index'] == j)
        worst = cases[np.argmax(assessment['relative_error'][cases])]
        cells.append(int(np.argmax(error[:, worst])))
    cells = np.unique(cells)
    rows = np.unique(ids[indices[cells]].ravel())
    return bank, rows, cells, mesh.cells[cells]


def refined_row(task):
    i, theta, level = task
    kw = dict(SOLVER_KW, t_end=900., max_snaps=6000)
    kw.update(LEVELS[level])
    values = np.full((3, len(R_OBS)), np.nan)
    valid = np.zeros(3, bool)
    try:
        result = solver.run_halo(sigma_m=10.**theta[0], w=10.**theta[1], **kw)
        status, ts = result['status'], result['t']
        if status not in (0, 1, 2):
            raise RuntimeError(f'numerical solver status {status}')
        for j, t in enumerate(TIMES):
            if t > ts[-1] + 1e-9:
                continue
            k = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
            alpha = np.clip((t - ts[k]) / (ts[k + 1] - ts[k]), 0, 1)
            ys = [np.interp(np.log10(R_OBS), np.log10(ds.shell_midpoints(result['r'][s])),
                            np.log10(result['rho'][s])) for s in (k, k + 1)]
            values[j] = (1 - alpha) * ys[0] + alpha * ys[1]
            valid[j] = True
            if not np.isfinite(values[j]).all():
                raise ValueError('nonfinite profile')
        return i, values, valid, status, result['t_final'], None
    except Exception as exc:
        return i, values, valid, -1, np.nan, repr(exc)


def diagnose(args):
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    bank, rows, cells, geometry = selected_cells()
    selection = dict(bank_rows=rows.tolist(), cells=cells.tolist(), geometry=geometry.tolist(),
                     bank_sha256=digest(F4 / 'bank.npz'),
                     protocol_sha256=digest(ROOT / 'docs/FORECAST_NUMERICS_PROTOCOL.md'))
    record = OUT / 'selection.json'
    if record.exists():
        assert json.loads(record.read_text()) == selection
    else:
        write_json(record, selection)
    base = {k: bank.d[k][rows] for k in ('params', 'profiles', 'valid', 'status', 't_end')}
    if not (DATA / 'level0.npz').exists():
        write_npz(DATA / 'level0.npz', **base)
    print('selected', len(cells), 'cells and', len(rows), 'unique nodes', flush=True)
    for level in LEVELS:
        path = DATA / f'level{level}.npz'
        meta = OUT / f'level{level}.json'
        if path.exists():
            assert digest(path) == json.loads(meta.read_text())['sha256']
            continue
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            result = list(pool.map(refined_row, [(i, p, level) for i, p in enumerate(base['params'])], chunksize=1))
        errors = [dict(index=r[0], error=r[-1]) for r in result if r[-1]]
        write_json(meta, dict(level=level, settings=LEVELS[level], failures=errors,
                             seconds=time.perf_counter() - start))
        if errors:
            raise RuntimeError(f'{len(errors)} numerical failures; see {meta}')
        write_npz(path, params=base['params'], profiles=np.stack([r[1] for r in result]),
                  valid=np.stack([r[2] for r in result]), status=np.array([r[3] for r in result]),
                  t_end=np.array([r[4] for r in result]))
        info = json.loads(meta.read_text())
        info['sha256'] = digest(path)
        write_json(meta, info)
        print('completed level', level, 'seconds', info['seconds'], flush=True)
    report = {}
    for a, b in ((0, 1), (1, 2), (0, 2)):
        first, second = np.load(DATA / f'level{a}.npz'), np.load(DATA / f'level{b}.npz')
        common = first['valid'] & second['valid']
        delta = second['profiles'] - first['profiles']
        mismatches = np.argwhere(first['valid'] != second['valid'])
        report[f'{a}_to_{b}'] = dict(
            max_profile_dex=float(np.max(np.abs(delta[common]))),
            rms_profile_dex=float(np.sqrt(np.mean(delta[common]**2))),
            support_disagreements=mismatches.tolist(),
            max_terminal_time_relative=float(np.max(np.abs(second['t_end'] / first['t_end'] - 1))),
            passed=bool(not len(mismatches) and np.max(np.abs(delta[common])) <= .1 * NOISE_DEX),
            times={str(int(t)): dict(n_common=int(common[:, j].sum()),
                max_profile_dex=float(np.max(np.abs(delta[common[:, j], j]))),
                rms_profile_dex=float(np.sqrt(np.mean(delta[common[:, j], j]**2))))
                for j, t in enumerate(TIMES)},
        )
    report['selection'] = selection
    report['source_sha256'] = digest(__file__)
    write_json(OUT / 'temporal_diagnostic.json', report)
    print(json.dumps({k: v for k, v in report.items() if '_to_' in k}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=20)
    diagnose(parser.parse_args())
