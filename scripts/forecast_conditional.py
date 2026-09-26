"""F6 conditional integration, tightly scoped to one difficult F5 observation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from forecast_numerics import refined_row, LEVELS
from forecast_calibrate import ROOT, LO, HI, NOISE_DEX, TIMES, write_npz
from forecast_validate import digest, write_json
from forecast_application import Ensemble, initial_nfw, interpolate
from forecast_integrate import summarize
from forecast_numerics_pilot import compare
from e9_inference import R_OBS
from gravop.conditional import ConditionalIntegral, outer_panel

DATA = ROOT / 'data/d1/forecast_conditional'
OUT = ROOT / 'results/forecast_conditional'
F5 = ROOT / 'data/d1/forecast_numerics'
CAP = 8000
TARGETS = [.02, .005, .00125]


def column_path(y):
    key = hashlib.sha256(np.float64(y).tobytes()).hexdigest()[:20]
    return DATA / 'columns' / key


def config():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    obs0 = np.load(F5 / 'pilot_observations.npz')
    obs = {k: v[2:3] for k, v in dict(obs0).items()}
    assert obs['time_index'][0] == 1
    expected = dict(case=353, time=400, settings=LEVELS[1],
        protocol_sha256=digest(ROOT / 'docs/FORECAST_CONDITIONAL_PROTOCOL.md'),
        inputs={name: digest(F5 / name) for name in ('pilot_bank.npz', 'pilot_truth.npz', 'pilot_observations.npz')},
        source_sha256={p: digest(ROOT / p) for p in
            ('src/gravop/conditional.py', 'scripts/forecast_conditional.py',
             'scripts/forecast_numerics.py', 'src/gravop/solver.py')},
        checkpoints={str((ROOT / 'results/forecast_validation/application' / f'sidm_full_s{s}/model.pt').relative_to(ROOT)):
                     digest(ROOT / 'results/forecast_validation/application' / f'sidm_full_s{s}/model.pt') for s in range(3)})
    path = OUT / 'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == expected, 'F6 inputs/settings/source changed; inspect before cache reuse'
    else:
        write_json(path, expected)
        write_npz(DATA / 'observation.npz', **obs)
    return expected, obs


class Column:
    def __init__(self, y, allowance):
        self.y, self.allowance = float(y), allowance
        self.path = column_path(y)
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        self.new_count = 0
        cache = self.path / 'nodes.npz'
        if cache.exists():
            d = np.load(cache)
            for i, x in enumerate(d['x']):
                self.rows[float(x)] = tuple(d[k][i] for k in ('profiles', 'valid', 'status', 't_end'))
            self.new_count = int(d['new_count'])
        else:
            for name in ('pilot_bank.npz', 'pilot_truth.npz'):
                d = np.load(F5 / name)
                for i in np.flatnonzero(d['params'][:, 1] == self.y):
                    self.rows[float(d['params'][i, 0])] = tuple(d[k][i] for k in ('profiles', 'valid', 'status', 't_end'))
        self.start_count = self.new_count
        self.predictions = {}
        self.ensemble = Ensemble('candidate', 'cpu')
        self.profile = initial_nfw()[None]
        self.obs = np.load(DATA / 'observation.npz')['observations'][0]

    def save(self):
        if not self.rows:
            return
        xs = np.array(sorted(self.rows))
        rows = [self.rows[x] for x in xs]
        write_npz(self.path / 'nodes.npz', x=xs, profiles=np.stack([r[0] for r in rows]),
            valid=np.stack([r[1] for r in rows]), status=np.array([r[2] for r in rows]),
            t_end=np.array([r[3] for r in rows]), new_count=self.new_count, y=self.y)

    def evaluate(self, x):
        x = float(x)
        if x not in self.rows:
            if self.new_count - self.start_count >= self.allowance or self.new_count >= 512:
                raise RuntimeError('conditional column new-solve cap reached')
            result = refined_row((0, np.array([x, self.y]), 1))
            if result[-1]:
                raise RuntimeError(result[-1])
            self.rows[x] = tuple(result[k] for k in (1, 2, 3, 4))
            self.new_count += 1
            if (self.new_count-self.start_count) % 4 == 0:
                self.save()
        if x not in self.predictions:
            y, ratio = self.ensemble.predict(self.profile, np.array([[x, self.y, 1.]]), 400.)
            self.predictions[x] = (interpolate(y[:, 0], R_OBS)[0], float(ratio[0]))
        profile, valid, status, end = self.rows[x]
        candidate, ratio = self.predictions[x]
        ok = np.array([valid[1], ratio <= 1])
        means = np.stack([profile[1], candidate])
        lp = np.full(2, -np.inf)
        lp[ok] = -.5*np.square((means[ok]-self.obs)/NOISE_DEX).sum(1)
        metric = np.array([end-400., 1.-ratio])
        # Match the documented snapshot endpoint tolerance exactly in support.
        metric = np.where(ok, np.maximum(metric, 1e-12), np.minimum(metric, -1e-12))
        return lp, metric


def run_column(task):
    y, stage, allowance = task
    torch.set_num_threads(1)
    obj = Column(y, allowance)
    obs = np.load(DATA / 'observation.npz')
    start = time.perf_counter()
    try:
        integrator = ConditionalIntegral(obj.evaluate, (LO[0], HI[0]), obs['params'][0, 0],
                                         obs['truth_log_likelihood'][0])
        result = integrator.integrate(TARGETS[stage]/10)
        obj.save()
        lp = np.stack([obj.evaluate(x)[0] for x in result['x']])
        write_npz(obj.path / f'stage{stage}.npz', **{k: result[k] for k in
            ('value', 'error', 'relative_error', 'intervals', 'x', 'weights')}, lp=lp, y=y)
        meta = dict(y=y, stage=stage, new_count=obj.new_count,
            additional_solves=obj.new_count-obj.start_count, converged=result['converged'],
            evaluations=result['evaluations'], discovery=result['discovery'],
            seconds=time.perf_counter()-start, sha256=digest(obj.path / f'stage{stage}.npz'))
        write_json(obj.path / f'stage{stage}.json', meta)
        return meta
    except Exception as exc:
        obj.save()
        meta = dict(y=y, stage=stage, new_count=obj.new_count, converged=False,
                    error=repr(exc), seconds=time.perf_counter()-start)
        write_json(obj.path / f'failure_stage{stage}.json', meta)
        return meta


def new_solves():
    return sum(int(np.load(path)['new_count']) for path in (DATA / 'columns').glob('*/nodes.npz'))


def get_columns(ys, stage, workers):
    columns = {}
    pending = []
    for y in sorted(set(map(float, ys))):
        path = column_path(y) / f'stage{stage}.npz'
        if path.exists():
            meta = json.loads(path.with_suffix('.json').read_text())
            assert digest(path) == meta['sha256'], 'conditional result cache changed'
            if not meta['converged']:
                raise RuntimeError('cached conditional result did not converge')
            columns[y] = dict(np.load(path))
        else:
            pending.append(y)
    while pending:
        remaining = CAP-new_solves()
        if remaining <= 0:
            raise RuntimeError('global new-solve cap reached')
        count = min(workers, len(pending), max(1, remaining//512))
        allowance = min(512, remaining//count)
        batch, pending = pending[:count], pending[count:]
        print('columns', batch, 'stage', stage, 'new solves so far', new_solves(), flush=True)
        with ProcessPoolExecutor(max_workers=count) as pool:
            for meta in pool.map(run_column, [(y, stage, allowance) for y in batch]):
                print('column complete', meta['y'], 'new', meta['new_count'], 'ok', meta['converged'],
                      meta.get('error', ''), flush=True)
                if not meta['converged']:
                    raise RuntimeError(f'conditional column failed: {meta}')
                columns[float(meta['y'])] = dict(np.load(column_path(meta['y']) / f'stage{stage}.npz'))
    return columns


def summarize_rule(intervals, panels, columns, obs):
    theta, weights, lps = [], [], []
    for panel in panels:
        for y, wy in zip(panel['y'], panel['weights']):
            column = columns[float(y)]
            theta.append(np.column_stack([column['x'], np.full(len(column['x']), y)]))
            weights.append(wy*column['weights'])
            lps.append(column['lp'])
    theta, weights, lp = np.concatenate(theta), np.concatenate(weights), np.concatenate(lps)
    value = sum(p['value'] for p in panels)
    estimates = {}
    for i, method in enumerate(('solver', 'candidate')):
        estimate = summarize(theta, weights, lp[:, i:i+1], lp[:, :1], obs)
        # CDF(w) is integrated with one-sided panel classification, not with
        # random ties attached to artificial atoms at outer Simpson endpoints.
        estimate['cdf'][0] = value[i, [3, 7, 4]]/value[i, 0]
        estimates[method] = estimate
    return estimates, dict(theta=theta, weights=weights, lp=lp, integrated=value, intervals=intervals)


def adaptive(args):
    frozen, obs = config()
    state_path = OUT / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('complete'):
            print('F6 already stopped; see progress.json', flush=True)
            return
        intervals, stage, history = np.array(state['intervals']), state['stage'], state['history']
    else:
        cuts = sorted({LO[1], .25, float(obs['params'][0, 1]), HI[1]})
        intervals, stage, history = np.column_stack([cuts[:-1], cuts[1:]]), 0, []
    passed, reason = False, 'in progress'
    try:
        while True:
            write_json(state_path, dict(intervals=intervals.tolist(), stage=stage, history=history, complete=False))
            ys = np.concatenate([a+(b-a)*np.arange(5)/4 for a, b in intervals])
            columns = get_columns(ys, stage, args.workers)
            panels = [outer_panel(a, b, columns, (LO[1], HI[1]), obs['params'][0, 1]) for a, b in intervals]
            value = sum(p['value'] for p in panels)
            error = sum(p['error'] for p in panels)
            if (value[:, 0] <= 0).any():
                raise RuntimeError('joint posterior has no sampled positive mass')
            relative = error/value[:, :1]
            row = dict(stage=stage, intervals=len(intervals), columns=len(columns),
                       new_solves=new_solves(), relative_error=relative.tolist())
            history.append(row)
            print('outer', row, flush=True)
            estimate, rule = summarize_rule(intervals, panels, columns, obs)
            write_npz(DATA / 'latest_rule.npz', **rule)
            for method, result in estimate.items():
                write_npz(OUT / f'latest_{method}.npz', **result)
            write_json(OUT / 'progress.json', dict(history=history, complete=False))
            if relative.max() <= TARGETS[stage]:
                write_npz(DATA / f'stage{stage}_rule.npz', **rule)
                for method, result in estimate.items():
                    write_npz(OUT / f'stage{stage}_{method}.npz', **result)
                if stage >= 1:
                    comparisons = {method: compare(np.load(OUT / f'stage{stage-1}_{method}.npz'), result)
                                   for method, result in estimate.items()}
                    write_json(OUT / 'convergence.json', comparisons)
                    if all(r['passed'] for r in comparisons.values()) or stage == 2:
                        passed = all(r['passed'] for r in comparisons.values())
                        reason = 'successive tolerance-level comparison'
                        break
                stage += 1
                intervals = np.array([part for a, b in intervals for part in ((a, (a+b)/2), ((a+b)/2, b))])
            else:
                score = [float(np.max(p['error']/value[:, :1])) for p in panels]
                i = int(np.argmax(score))
                a, b = intervals[i]
                if b-a < 1e-6:
                    raise RuntimeError('outer resolution limit reached')
                intervals = np.concatenate([intervals[:i], [[a, (a+b)/2], [(a+b)/2, b]], intervals[i+1:]])
    except Exception as exc:
        reason = repr(exc)
        print('F6 stop', reason, flush=True)
    write_json(state_path, dict(intervals=intervals.tolist(), stage=stage, history=history, complete=True))
    write_json(OUT / 'progress.json', dict(history=history, complete=True, numerical_passed=passed,
               reason=reason, new_solves=new_solves(), final_stage=stage))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    adaptive(parser.parse_args())
