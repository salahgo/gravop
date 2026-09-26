"""Resumable conditional integration and stable adaptive outer panels."""
import json
from pathlib import Path
import threading

import numpy as np

from .conditional import ConditionalIntegral
from .dyadic import dyadic_nodes


class RefinedConditional(ConditionalIntegral):
    """Batch independent nodes; retain a verified earlier interval partition."""
    def __init__(self, *args, prefetch=None, warm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch = prefetch or (lambda xs: None)
        self.warm = warm

    def discover(self):
        if self.warm is None:
            self.prefetch(np.linspace(self.lo, self.hi, 33))
            return super().discover()
        intervals = np.asarray(self.warm['intervals'], float).copy()
        if (intervals.ndim != 2 or intervals.shape[1] != 2 or
                not np.isfinite(intervals).all() or
                not (intervals[:, 1] > intervals[:, 0]).all() or
                intervals[0, 0] != self.lo or intervals[-1, 1] != self.hi or
                not np.array_equal(intervals[:-1, 1], intervals[1:, 0])):
            raise ValueError('warm partition must cover the complete original bounds')
        self.discovery = self.warm['discovery']
        return intervals

    def panel(self, a, b):
        xs = np.concatenate([(a+b)/2+(b-a)/2*z for z, _ in self.rules.values()])
        self.prefetch(xs)
        return super().panel(a, b)


class OuterMesh:
    """Fixed base bounds and integer descendants; no rounded-child node keys."""
    def __init__(self, roots, cells=None):
        self.roots = np.asarray(roots, float)
        if (self.roots.ndim != 2 or self.roots.shape[1] != 2 or
                not np.isfinite(self.roots).all() or
                not (self.roots[:, 1] > self.roots[:, 0]).all() or
                not np.array_equal(self.roots[:-1, 1], self.roots[1:, 0])):
            raise ValueError('roots must partition one finite interval')
        self.cells = [tuple(map(int, c)) for c in cells] if cells is not None else [
            (i, 0, 0) for i in range(len(self.roots))]
        spans = self.intervals()
        if (spans[0, 0] != self.roots[0, 0] or spans[-1, 1] != self.roots[-1, 1] or
                not np.array_equal(spans[:-1, 1], spans[1:, 0])):
            raise ValueError('cells must partition the original roots')

    def nodes(self, cell):
        root, index, depth = cell
        if not 0 <= root < len(self.roots):
            raise ValueError('invalid base root')
        return dyadic_nodes(self.roots[root], index, depth)

    def intervals(self):
        return np.array([self.nodes(c)[[0, -1]] for c in self.cells])

    def required(self):
        return sorted(set(float(y) for c in self.cells for y in self.nodes(c)))

    def refine(self, indices):
        selected = set(indices)
        self.cells = [child for k, (root, index, depth) in enumerate(self.cells)
            for child in ([(root, 2*index, depth+1), (root, 2*index+1, depth+1)]
                          if k in selected else [(root, index, depth)])]

    def localize(self, brackets):
        """Split only branches overlapping a detected feature bracket."""
        for a, b in brackets:
            if b-a < 1e-12:
                continue
            while True:
                selected = [i for i, (l, h) in enumerate(self.intervals())
                            if l < b-1e-14 and h > a+1e-14 and h-l > (b-a)*(1+1e-12)]
                if not selected:
                    break
                self.refine(selected)

    def assess(self, columns, truth_y):
        if not np.any(self.intervals() == truth_y):
            raise ValueError('outer partition must split at the truth CDF boundary')
        lo, hi = self.roots[0, 0], self.roots[-1, 1]
        panels = []
        for cell in self.cells:
            y = self.nodes(cell)
            values = np.stack([columns[float(v)]['value'] for v in y])
            errors = np.stack([columns[float(v)]['error'] for v in y])
            u = (y-lo)/(hi-lo)
            side = float((y[0]+y[-1])/2 < truth_y)
            def augment(v):
                return np.concatenate([v, v[:, :, :1]*u[:, None, None],
                    v[:, :, :1]*u[:, None, None]**2, v[:, :, :1]*side], axis=2)
            quantities, uncertainties = augment(values), augment(errors)
            span = y[-1]-y[0]
            weights = span*np.array([1, 4, 2, 4, 1])/12
            fine = np.einsum('n,nmk->mk', weights, quantities)
            coarse = span*np.einsum('n,nmk->mk', np.array([1, 4, 1])/6, quantities[::2])
            propagated = np.einsum('n,nmk->mk', weights, uncertainties)
            panels.append(dict(value=fine, error=abs(fine-coarse)+propagated,
                inner_error=propagated, y=y, weights=weights))
        return panels


def feature_brackets(columns, cdf_jump=.2, log_mass_jump=4.):
    """Locate observed CDF/evidence changes; no removal of low-mass regions."""
    ys = sorted(columns)
    maximum = np.max([columns[y]['value'][:, 0] for y in ys], axis=0)
    found = []
    for a, b in zip(ys[:-1], ys[1:]):
        if b-a < 1e-12:
            continue
        va, vb = columns[a]['value'], columns[b]['value']
        za, zb = va[:, 0], vb[:, 0]
        relevant = np.maximum(za, zb) > 1e-4*maximum
        ca = va[:, [3, 4]]/np.maximum(za[:, None], 1e-300)
        cb = vb[:, [3, 4]]/np.maximum(zb[:, None], 1e-300)
        jump = (abs(ca-cb).max(1) >= cdf_jump) | (
            abs(np.log(np.maximum(za, 1e-300))-np.log(np.maximum(zb, 1e-300))) >= log_mass_jump)
        if (relevant & jump).any():
            found.append((a, b))
    return found


class SolveBudget:
    """Durable reservation before submission, including interrupted attempts."""
    def __init__(self, path, limit, per_column):
        self.path = Path(path)
        self.limit, self.per_column = limit, per_column
        self.lock = threading.Lock()
        self.counts = {}
        if self.path.exists():
            d = json.loads(self.path.read_text())
            if d['limit'] != limit or d['per_column'] != per_column:
                raise ValueError('saved solver budget differs')
            self.counts = d['columns']

    @property
    def used(self):
        return sum(self.counts.values())

    def reserve(self, key, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError('reservation must be a nonnegative integer')
        with self.lock:
            if self.used+n > self.limit:
                raise RuntimeError('global solver submission budget exhausted')
            if self.counts.get(key, 0)+n > self.per_column:
                raise RuntimeError('conditional solver submission budget exhausted')
            self.counts[key] = self.counts.get(key, 0)+n
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(limit=self.limit, per_column=self.per_column,
                                           columns=self.counts), indent=2))
            tmp.replace(self.path)
