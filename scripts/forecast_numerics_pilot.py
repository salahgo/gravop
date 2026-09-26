"""F5 bounded higher-order integration on consistently tighter solver profiles."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time

import numpy as np
import torch

from forecast_integrate import log_likelihood, summarize, DyadicMesh, Bank
from forecast_numerics import ROOT, DATA, OUT, F4, refined_row
from forecast_calibrate import LO, HI, TIMES, write_npz
from forecast_validate import digest, write_json
from forecast_application import Ensemble, initial_nfw, interpolate
from e9_inference import R_OBS
from gravop.simpson import SimpsonMesh

CAP = 4096
LATTICE = 65536


def solve(params, workers, level=1):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(refined_row, [(i, p, level) for i, p in enumerate(params)], chunksize=1))
    errors = [r[-1] for r in rows if r[-1]]
    if errors:
        raise RuntimeError(errors)
    return dict(params=params, profiles=np.stack([r[1] for r in rows]),
                valid=np.stack([r[2] for r in rows]), status=np.array([r[3] for r in rows]),
                t_end=np.array([r[4] for r in rows]))


class PilotBank:
    def __init__(self):
        path = DATA / 'pilot_bank.npz'
        origin = path if path.exists() else DATA / 'level1.npz'
        with np.load(origin) as f:
            self.d = {k: f[k] for k in f.files}
        if 'keys' not in self.d:
            self.d['keys'] = np.rint((self.d['params'] - LO) / (HI - LO) * LATTICE).astype(int)
            write_npz(path, **self.d)
        self.index = {tuple(k): i for i, k in enumerate(self.d['keys'])}

    def ids(self, keys):
        return np.array([self.index[tuple(k)] for k in keys])

    def ensure(self, keys, workers):
        missing = np.array([k for k in keys if tuple(k) not in self.index], dtype=int)
        if not len(missing):
            return True
        if len(self.index) + len(missing) > CAP:
            return False
        print('solve level-1 nodes', len(missing), 'total', len(self.index) + len(missing), flush=True)
        new = solve(LO + missing / LATTICE * (HI - LO), workers)
        new['keys'] = missing
        self.d = {k: np.concatenate([self.d[k], new[k]]) for k in self.d}
        self.index = {tuple(k): i for i, k in enumerate(self.d['keys'])}
        write_npz(DATA / 'pilot_bank.npz', **self.d)
        return True


def prepare(args):
    record = OUT / 'pilot_selection.json'
    if record.exists():
        selected = json.loads(record.read_text())
        assert selected['protocol_sha256'] == digest(ROOT / 'docs/FORECAST_NUMERICS_PILOT.md')
        assert selected['f4_observations_sha256'] == digest(F4 / 'observations.npz'), 'F4 observations changed'
        assert selected['level1_sha256'] == digest(DATA / 'level1.npz'), 'level-1 reference changed'
    else:
        f4 = Bank()
        obs = np.load(F4 / 'observations.npz')
        mesh = DyadicMesh(cells=json.loads((F4 / 'mesh.json').read_text())['cells'])
        keys, ix = mesh.stencil()
        lp = log_likelihood(f4.d['profiles'], f4.d['valid'], obs)
        assessment = mesh.assess(np.exp(lp[f4.ids(keys)] - lp.max(0)), ix)
        cases = []
        for j in range(3):
            rows = np.flatnonzero(obs['time_index'] == j)
            order = rows[np.argsort(assessment['relative_error'][rows])]
            cases.extend([int(order[-1]), int(order[len(order) // 2])])
        seeds = (f4.d['keys'][np.argmax(lp[:, cases], axis=0)] * 16).tolist()
        selected = dict(cases=cases, seed_keys=seeds,
                        protocol_sha256=digest(ROOT / 'docs/FORECAST_NUMERICS_PILOT.md'),
                        f4_observations_sha256=digest(F4 / 'observations.npz'),
                        level1_sha256=digest(DATA / 'level1.npz'))
        write_json(record, selected)
    path = DATA / 'pilot_observations.npz'
    if not path.exists():
        old = np.load(F4 / 'observations.npz')
        obs = {k: old[k][selected['cases']] for k in old.files}
        truth = solve(obs['params'], args.workers)
        valid = truth['valid'][np.arange(len(obs['row'])), obs['time_index']]
        if not valid.all():
            write_npz(DATA / 'pilot_truth.npz', **truth)
            raise RuntimeError('level-1 truth support changed; retain cases and investigate')
        from forecast_calibrate import NOISE_DEX
        delta = obs['observations'] - truth['profiles'][np.arange(len(valid)), obs['time_index']]
        obs['truth_log_likelihood'] = -.5 * np.square(delta / NOISE_DEX).sum(1)
        write_npz(DATA / 'pilot_truth.npz', **truth)
        write_npz(path, **obs)
    return selected, np.load(path)


def compare(a, b):
    mean = np.abs(a['mean'] - b['mean']) / np.maximum(b['sd'], 1e-12)
    width = np.abs(a['width90'] - b['width90']) / np.maximum(b['width90'], 1e-12)
    cdf = np.abs(a['cdf'] - b['cdf'])
    return dict(mean_shift_sd=mean.tolist(), relative_width_change=width.tolist(),
                cdf_change=cdf.tolist(),
                passed=bool((mean <= .1).all() and (width <= .1).all() and (cdf <= .02).all()))


def seeded_mesh(selected):
    mesh = SimpsonMesh(base=8)
    for point in selected['seed_keys']:
        point = np.asarray(point)
        while True:
            # On cell boundaries either adjacent cell can supply the node.
            inside = np.all((point >= mesh.cells[:, :2]) &
                            (point <= mesh.cells[:, :2] + mesh.cells[:, 2, None]), axis=1)
            candidates = np.flatnonzero(inside)
            if any(np.all((point - mesh.cells[c, :2]) % (mesh.cells[c, 2] // 4) == 0)
                   for c in candidates):
                break
            mesh.refine([candidates[0]])
    return mesh


def adaptive(args):
    selection, obs = prepare(args)
    bank = PilotBank()
    state_path = DATA / 'pilot_mesh.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('complete') and not args.refine_next:
            print('pilot already stopped; see pilot_progress.json', flush=True)
            return
        mesh = SimpsonMesh(cells=state['cells'])
        stage, history = state['stage'], state['history']
        if args.refine_next:
            progress = json.loads((OUT / 'pilot_progress.json').read_text())
            if not state.get('complete') or progress['reason'] != 'successive tolerance-level check' or stage >= 2:
                raise ValueError('fallback requires completed tolerance levels and an unused third stage')
            stage += 1
    else:
        mesh, stage, history = seeded_mesh(selection), 0, []
        write_json(state_path, dict(cells=mesh.cells.tolist(), stage=stage, history=history))
    targets = [.02, .005, .00125]
    start = time.perf_counter()
    converged = False
    reason = 'node or resolution cap'
    while True:
        keys, ix = mesh.stencil()
        if not bank.ensure(keys, args.workers):
            break
        ids = bank.ids(keys)
        lp = log_likelihood(bank.d['profiles'], bank.d['valid'], obs)
        assessment = mesh.assess(np.exp(lp[ids] - lp.max(0)), ix)
        weights = np.zeros(len(bank.d['keys']))
        weights[ids] = mesh.weights(ix, len(keys))
        label = 'pilot_initial' if not history else 'pilot_latest'
        write_npz(DATA / f'{label}.npz', weights=weights, keys=bank.d['keys'])
        write_npz(DATA / 'pilot_latest.npz', weights=weights, keys=bank.d['keys'])
        row = dict(iteration=len(history), nodes=len(weights), cells=len(mesh.cells),
                   relative_errors=assessment['relative_error'].tolist(), target=targets[stage],
                   seconds_this_session=time.perf_counter() - start)
        history.append(row)
        print('pilot', row, flush=True)
        write_json(OUT / 'pilot_progress.json', dict(history=history, complete=False))
        if assessment['relative_error'].max() <= targets[stage]:
            estimate = summarize(bank.d['params'], weights, lp, lp, obs)
            write_npz(DATA / f'pilot_stage{stage}.npz', weights=weights, keys=bank.d['keys'])
            write_npz(DATA / f'pilot_summary{stage}.npz', **estimate)
            if stage >= 1:
                comp = compare(np.load(DATA / f'pilot_summary{stage-1}.npz'), estimate)
                write_json(OUT / 'pilot_solver_convergence.json', comp)
                if comp['passed'] or stage == 2:
                    converged = comp['passed']
                    reason = 'successive tolerance-level check'
                    break
            stage += 1
        eligible = np.flatnonzero(mesh.cells[:, 2] >= 8)
        rank = eligible[np.argsort(assessment['priority'][eligible])[::-1]]
        count = min(16, (CAP - len(weights)) // 56, len(rank))
        if not count:
            break
        mesh.refine(rank[:count])
        write_json(state_path, dict(cells=mesh.cells.tolist(), stage=stage, history=history))
    write_json(state_path, dict(cells=mesh.cells.tolist(), stage=stage, history=history, complete=True))
    write_json(OUT / 'pilot_progress.json', dict(history=history, complete=True,
               converged=converged, reason=reason, final_stage=stage))
    print('pilot stop', reason, 'converged', converged, flush=True)


def evaluate(args):
    selection, obs = prepare(args)
    bank = PilotBank()
    progress = json.loads((OUT / 'pilot_progress.json').read_text())
    assert progress['complete']
    stages = sorted(DATA.glob('pilot_stage[0-9].npz'))
    if progress['reason'] == 'successive tolerance-level check':
        rules = stages[-2:]
    else:
        rules = (stages[-1:] or [DATA / 'pilot_initial.npz']) + [DATA / 'pilot_latest.npz']
    lp = log_likelihood(bank.d['profiles'], bank.d['valid'], obs)
    ens = Ensemble('candidate', args.device)
    profile = initial_nfw()
    par = np.column_stack([bank.d['params'], np.ones(len(bank.d['params']))])
    ys, valid = [], []
    for t in TIMES:
        y, ok = [], []
        for off in range(0, len(par), 256):
            p = par[off:off+256]
            output, q = ens.predict(np.repeat(profile[None], len(p), axis=0), p, t)
            if not np.isfinite(output).all() or not np.isfinite(q).all():
                raise ValueError('nonfinite candidate prediction')
            y.append(interpolate(output[:, 0], R_OBS))
            ok.append(q <= 1)
        ys.append(np.concatenate(y))
        valid.append(np.concatenate(ok))
    predicted_profiles, predicted_valid = np.stack(ys, 1), np.stack(valid, 1)
    candidate = log_likelihood(predicted_profiles, predicted_valid, obs)
    write_npz(DATA / 'pilot_candidate_predictions.npz', profiles=predicted_profiles, valid=predicted_valid)
    out = dict(selection=selection, progress=progress, rules=[p.name for p in rules],
               convergence={}, checkpoint_hashes=ens.hashes)
    final = {}
    for kind, values in (('solver', lp), ('candidate', candidate)):
        estimates = []
        for path in rules:
            rule = np.load(path)
            np.testing.assert_array_equal(rule['keys'], bank.d['keys'][:len(rule['keys'])])
            weights = np.pad(rule['weights'], (0, len(lp) - len(rule['weights'])))
            estimates.append(summarize(bank.d['params'], weights, values, lp, obs))
        out['convergence'][kind] = compare(*estimates)
        final.update({kind + '_' + k: v for k, v in estimates[-1].items()})
    # The pilot protocol requires a targeted level-2 check at newly found modes.
    modes = np.unique(np.argmax(np.where(weights[:, None] > 0, lp, -np.inf), axis=0))
    path = DATA / 'pilot_mode_refinement.npz'
    if path.exists():
        strict = dict(np.load(path))
        np.testing.assert_array_equal(strict['params'], bank.d['params'][modes])
    else:
        strict = solve(bank.d['params'][modes], args.workers, level=2)
        write_npz(path, **strict)
    common = strict['valid'] & bank.d['valid'][modes]
    delta = strict['profiles'] - bank.d['profiles'][modes]
    maximum = float(np.max(np.abs(delta[common])))
    mismatches = np.argwhere(strict['valid'] != bank.d['valid'][modes]).tolist()
    out['mode_temporal_check'] = dict(bank_rows=modes.tolist(), max_profile_dex=maximum,
        support_disagreements=mismatches, passed=bool(not mismatches and maximum <= .00434))
    out['pilot_passed'] = bool(progress['converged'] and
        all(v['passed'] for v in out['convergence'].values()) and out['mode_temporal_check']['passed'])
    out['bank_sha256'] = digest(DATA / 'pilot_bank.npz')
    out['observations_sha256'] = digest(DATA / 'pilot_observations.npz')
    out['predictions_sha256'] = digest(DATA / 'pilot_candidate_predictions.npz')
    out['source_sha256'] = {p: digest(ROOT / p) for p in
        ('scripts/forecast_numerics.py', 'scripts/forecast_numerics_pilot.py', 'src/gravop/simpson.py')}
    write_npz(OUT / 'pilot_diagnostics.npz', **final)
    write_json(OUT / 'pilot_validation.json', out)
    print('pilot passed', out['pilot_passed'], 'temporal check', out['mode_temporal_check'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['adaptive', 'evaluate'])
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--refine-next', action='store_true', help='use the predefined third tolerance after candidate convergence failure')
    args = parser.parse_args()
    torch.set_num_threads(2)
    globals()[args.stage](args)
