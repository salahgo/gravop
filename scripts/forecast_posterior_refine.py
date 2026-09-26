"""F7: feature-focused outer refinement with durable, shared solver work."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

import forecast_conditional as f6
from forecast_application import Ensemble, initial_nfw, interpolate
from forecast_calibrate import ROOT, LO, HI, NOISE_DEX, write_npz
from forecast_numerics import refined_row
from forecast_numerics_pilot import compare
from forecast_validate import digest, write_json
from e9_inference import R_OBS
from gravop.conditional_refinement import RefinedConditional, OuterMesh, feature_brackets, SolveBudget

DATA = ROOT/'data/d1/forecast_posterior_refine'
OUT = ROOT/'results/forecast_posterior_refine'
TARGETS = [.02, .005, .00125]
CAP, COLUMN_CAP = 80000, 1024


def arrays(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def key(y):
    return hashlib.sha256(np.float64(y).tobytes()).hexdigest()[:20]


def column_path(y):
    return DATA/'columns'/key(y)


def prepare():
    base, obs = f6.config()
    manifest_path = f6.OUT/'data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        assert digest(ROOT/name) == expected, f'F6 data changed: {name}'
    old = arrays(f6.DATA/'latest_rule.npz')
    cuts = [-1., .25, float(obs['params'][0, 1]), 1.5]
    roots = np.column_stack([cuts[:-1], cuts[1:]])
    cells = []
    for a, b in old['intervals']:
        root = next(i for i, (l, h) in enumerate(roots) if a >= l-1e-14 and b <= h+1e-14)
        l, h = roots[root]
        depth = int(round(np.log2((h-l)/(b-a))))
        index = int(round((a-l)/(h-l)*(1 << depth)))
        cells.append((root, index, depth))
    mesh = OuterMesh(roots, cells)
    np.testing.assert_allclose(mesh.intervals(), old['intervals'], atol=1e-14, rtol=0)
    old_columns, meta_hashes = {}, {}
    for path in (f6.DATA/'columns').glob('*/stage0.json'):
        meta = json.loads(path.read_text())
        assert digest(path.with_suffix('.npz')) == meta['sha256']
        meta_hashes[str(path.relative_to(ROOT))] = digest(path)
        if meta['converged']:
            old_columns[float(meta['y'])] = arrays(path.with_suffix('.npz'))
    brackets = feature_brackets(old_columns)
    mesh.localize(brackets)
    expected = dict(case=353, time=400, base_f6=base, base_data_manifest_sha256=digest(manifest_path),
        base_column_metadata=meta_hashes, targets=TARGETS, new_submission_cap=CAP,
        per_column_submission_cap=COLUMN_CAP, roots=roots.tolist(), initial_cells=mesh.cells,
        feature_brackets=brackets, protocol_sha256=digest(ROOT/'docs/FORECAST_POSTERIOR_REFINE_PROTOCOL.md'),
        source_sha256={name: digest(ROOT/name) for name in ('scripts/forecast_posterior_refine.py',
            'src/gravop/conditional_refinement.py', 'src/gravop/dyadic.py')})
    expected = json.loads(json.dumps(expected))
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    path = OUT/'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == expected, 'F7 configuration/source changed'
        np.testing.assert_array_equal(arrays(DATA/'observation.npz')['observations'], obs['observations'])
    else:
        write_json(path, expected)
        write_npz(DATA/'observation.npz', **obs)
    return expected, obs


def worker_init():
    torch.set_num_threads(1)


class Column:
    def __init__(self, y, context):
        self.y, self.context = float(y), context
        self.path = column_path(y)
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows, self.predictions = {}, {}
        self.new_count = 0
        cache = self.path/'nodes.npz'
        inherited = f6.column_path(y)/'nodes.npz'
        if cache.exists() or inherited.exists():
            d = arrays(cache if cache.exists() else inherited)
            assert float(d['y']) == self.y
            self.new_count = int(d['new_count']) if cache.exists() else 0
            for i, x in enumerate(d['x']):
                self.rows[float(x)] = tuple(d[k][i] for k in ('profiles', 'valid', 'status', 't_end'))
        else:
            for name in ('pilot_bank.npz', 'pilot_truth.npz'):
                d = context.seeds[name]
                for i in np.flatnonzero(d['params'][:, 1] == self.y):
                    self.rows[float(d['params'][i, 0])] = tuple(d[k][i] for k in ('profiles', 'valid', 'status', 't_end'))

    def save(self):
        if not self.rows:
            return
        xs = np.array(sorted(self.rows))
        rows = [self.rows[x] for x in xs]
        write_npz(self.path/'nodes.npz', x=xs, profiles=np.stack([r[0] for r in rows]),
            valid=np.stack([r[1] for r in rows]), status=np.array([r[2] for r in rows]),
            t_end=np.array([r[3] for r in rows]), new_count=self.new_count, y=self.y)

    def prefetch(self, xs):
        missing = [x for x in dict.fromkeys(map(float, xs)) if x not in self.rows]
        if not missing:
            return
        self.context.budget.reserve(key(self.y), len(missing))
        futures = {self.context.pool.submit(refined_row, (i, np.array([x, self.y]), 1)): x
                   for i, x in enumerate(missing)}
        failures = []
        for future in as_completed(futures):
            x = futures[future]
            try:
                result = future.result()
                self.new_count += 1
                if result[-1]:
                    failures.append(dict(x=x, error=result[-1]))
                else:
                    self.rows[x] = tuple(result[k] for k in (1, 2, 3, 4))
            except Exception as exc:
                failures.append(dict(x=x, error=repr(exc)))
            # Persist completed work even if another node in this batch fails.
            self.save()
        if failures:
            write_json(self.path/'solver_failures.json', failures)
            raise RuntimeError(f'solver failures: {failures}')

    def evaluate(self, x):
        x = float(x)
        self.prefetch([x])
        if x not in self.predictions:
            fields, ratio = self.context.ensemble.predict(self.context.profile,
                np.array([[x, self.y, 1.]]), 400.)
            self.predictions[x] = (interpolate(fields[:, 0], R_OBS)[0], float(ratio[0]))
        profile, valid, status, end = self.rows[x]
        candidate, ratio = self.predictions[x]
        ok = np.array([valid[1], ratio <= 1])
        means = np.stack([profile[1], candidate])
        lp = np.full(2, -np.inf)
        lp[ok] = -.5*np.square((means[ok]-self.context.obs['observations'][0])/NOISE_DEX).sum(1)
        metric = np.array([end-400., 1.-ratio])
        metric = np.where(ok, np.maximum(metric, 1e-12), np.minimum(metric, -1e-12))
        return lp, metric


class Context:
    def __init__(self, obs, pool):
        self.obs, self.pool = obs, pool
        self.budget = SolveBudget(OUT/'budget.json', CAP, COLUMN_CAP)
        self.ensemble = Ensemble('candidate', 'cpu')
        self.profile = initial_nfw()[None]
        self.seeds = {name: arrays(f6.F5/name) for name in ('pilot_bank.npz', 'pilot_truth.npz')}

    def run_column(self, task):
        y, stage = task
        path = column_path(y)/f'stage{stage}.npz'
        if path.exists():
            meta = json.loads(path.with_suffix('.json').read_text())
            assert digest(path) == meta['sha256']
            if meta['converged']:
                return float(y), arrays(path)
        start = time.perf_counter()
        obj = Column(y, self)
        warm = None
        # The observation, solver, candidate and all original cuts are unchanged.
        candidates = [obj.path/f'stage{s}.npz' for s in range(stage, -1, -1)]
        candidates.append(f6.column_path(y)/'stage0.npz')
        for prior in candidates:
            if prior.exists():
                meta = json.loads(prior.with_suffix('.json').read_text())
                assert digest(prior) == meta['sha256']
                old = arrays(prior)
                warm = dict(intervals=old['intervals'], discovery=meta['discovery'])
                break
        try:
            integrator = RefinedConditional(obj.evaluate, (LO[0], HI[0]),
                self.obs['params'][0, 0], self.obs['truth_log_likelihood'][0],
                prefetch=obj.prefetch, warm=warm)
            result = integrator.integrate(TARGETS[stage]/10)
            obj.save()
            lp = np.stack([obj.evaluate(x)[0] for x in result['x']])
            saved = {k: result[k] for k in ('value', 'error', 'relative_error', 'intervals', 'x', 'weights')}
            saved.update(lp=lp, y=y)
            write_npz(path, **saved)
            meta = dict(y=y, stage=stage, converged=result['converged'], new_count=obj.new_count,
                discovery=result['discovery'], seconds=time.perf_counter()-start,
                reused_partition=warm is not None, sha256=digest(path))
            write_json(path.with_suffix('.json'), meta)
            print('column', y, 'stage', stage, 'new', obj.new_count, 'pass', result['converged'], flush=True)
            if not result['converged']:
                raise RuntimeError('conditional quadrature did not converge')
            return float(y), saved
        except Exception as exc:
            obj.save()
            write_json(obj.path/f'failure_stage{stage}.json', dict(y=y, stage=stage,
                       error=repr(exc), new_count=obj.new_count))
            raise


def adaptive(args):
    frozen, obs = prepare()
    path = OUT/'state.json'
    if path.exists():
        state = json.loads(path.read_text())
        if state.get('complete'):
            print('F7 already stopped; see progress.json')
            return
        mesh = OuterMesh(frozen['roots'], state['cells'])
        stage, history = state['stage'], state['history']
    else:
        mesh = OuterMesh(frozen['roots'], frozen['initial_cells'])
        stage, history = 0, []
    passed, reason = False, 'in progress'
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        context = Context(obs, pool)
        try:
            while True:
                write_json(path, dict(cells=mesh.cells, stage=stage, history=history, complete=False))
                ys = mesh.required()
                print('outer request', 'stage', stage, 'panels', len(mesh.cells), 'columns', len(ys),
                      'submissions', context.budget.used, flush=True)
                with ThreadPoolExecutor(max_workers=args.columns) as threads:
                    columns = dict(threads.map(context.run_column, [(y, stage) for y in ys]))
                panels = mesh.assess(columns, obs['params'][0, 1])
                value = sum(p['value'] for p in panels)
                error = sum(p['error'] for p in panels)
                if not np.isfinite(value).all() or (value[:, 0] <= 0).any():
                    raise RuntimeError('invalid joint posterior mass')
                relative = error/value[:, :1]
                row = dict(stage=stage, cells=mesh.cells, panels=len(mesh.cells), columns=len(ys),
                           submissions=context.budget.used, relative_error=relative.tolist())
                history.append(row)
                print('outer', row, flush=True)
                estimates, rule = f6.summarize_rule(mesh.intervals(), panels, columns, obs)
                rule.update(roots=mesh.roots, cells=np.array(mesh.cells), error=error, stage=stage)
                write_npz(DATA/'latest_rule.npz', **rule)
                for method, result in estimates.items():
                    write_npz(OUT/f'latest_{method}.npz', **result)
                write_json(OUT/'progress.json', dict(history=history, complete=False))
                if relative.max() <= TARGETS[stage]:
                    write_npz(DATA/f'stage{stage}_rule.npz', **rule)
                    for method, result in estimates.items():
                        write_npz(OUT/f'stage{stage}_{method}.npz', **result)
                    if stage >= 1:
                        comparisons = {method: compare(arrays(OUT/f'stage{stage-1}_{method}.npz'), result)
                                       for method, result in estimates.items()}
                        write_json(OUT/'convergence.json', comparisons)
                        if all(r['passed'] for r in comparisons.values()) or stage == 2:
                            passed = all(r['passed'] for r in comparisons.values())
                            reason = 'successive tolerance-level comparison'
                            break
                    stage += 1
                    mesh.refine(range(len(mesh.cells)))
                else:
                    priority = [np.max(p['error']/value[:, :1]) for p in panels]
                    mesh.refine([int(np.argmax(priority))])
        except Exception as exc:
            reason = repr(exc)
            print('F7 stop', reason, flush=True)
        write_json(path, dict(cells=mesh.cells, stage=stage, history=history, complete=True))
        write_json(OUT/'progress.json', dict(history=history, complete=True, numerical_passed=passed,
                   reason=reason, submissions=context.budget.used, final_stage=stage))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--columns', type=int, default=12)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            prepare()
        else:
            adaptive(args)
