"""Positive conditional quadrature with explicit support and CDF breakpoints.

The callback returns log likelihoods and signed support functions for each
method. Global surveys and local optimizations locate candidate modes, not a
proof that every possible narrow mode has been found.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize_scalar


class ConditionalIntegral:
    def __init__(self, evaluate, bounds, truth_x, truth_loglike, reference=0):
        self.evaluate = evaluate
        self.lo, self.hi = map(float, bounds)
        self.truth_x = float(truth_x)
        self.truth_loglike = float(truth_loglike)
        self.reference = reference
        self.cache = {}
        self.rules = {n: leggauss(n) for n in (3, 5)}

    def at(self, x):
        x = float(x)
        if x not in self.cache:
            lp, support = self.evaluate(x)
            lp, support = np.asarray(lp, float), np.asarray(support, float)
            if lp.ndim != 1 or lp.shape != support.shape:
                raise ValueError('likelihood/support vectors must agree')
            if np.isnan(lp).any() or np.isposinf(lp).any() or not np.isfinite(support).all():
                raise ValueError('invalid likelihood or support function')
            if (lp[support < 0] != -np.inf).any():
                raise ValueError('likelihood must vanish outside its support')
            self.cache[x] = (lp, support)
        return self.cache[x]

    def discover(self):
        grid = np.linspace(self.lo, self.hi, 33)
        points = set(grid.tolist())
        boundaries = {self.lo, self.hi}
        if self.lo < self.truth_x < self.hi:
            boundaries.add(self.truth_x)
        support_roots = []
        values = [self.at(x) for x in grid]
        n_methods = len(values[0][0])
        for method in range(n_methods):
            for i in range(len(grid) - 1):
                a, b = values[i][1][method], values[i + 1][1][method]
                if a * b < 0:
                    root = brentq(lambda x: self.at(x)[1][method], grid[i], grid[i+1], xtol=1e-7)
                    boundaries.add(root)
                    points.add(root)
                    support_roots.append(dict(method=method, x=root))
                elif a == 0:
                    boundaries.add(float(grid[i]))
        modes = []
        for method in range(n_methods):
            xs = sorted(points)
            lp = np.array([self.at(x)[0][method] for x in xs])
            peaks = [i for i in range(len(xs)) if np.isfinite(lp[i]) and
                     (i == 0 or lp[i] >= lp[i-1]) and (i == len(xs)-1 or lp[i] >= lp[i+1])]
            # Exactly flat conditional likelihoods need no repeated optimization.
            if len(peaks) > 12 and np.ptp(lp[np.isfinite(lp)]) < 1e-10:
                peaks = [peaks[len(peaks)//2]]
            if len(peaks) > 12:
                raise ValueError('more than twelve candidate modes; inspect conditional shape')
            for i in peaks:
                left, right = xs[max(i-1, 0)], xs[min(i+1, len(xs)-1)]
                optimum = minimize_scalar(lambda x: -max(self.at(x)[0][method], -1e100),
                    bounds=(left, right), method='bounded', options=dict(xatol=1e-7, maxiter=80))
                if not optimum.success:
                    raise RuntimeError('conditional mode optimization failed')
                peak = max([xs[i], float(optimum.x)], key=lambda x: self.at(x)[0][method])
                points.add(peak)
                boundaries.add(peak)
                modes.append(dict(method=method, x=peak, loglike=float(self.at(peak)[0][method])))
                # Locate shoulders using a likelihood drop, rather than assuming
                # an ellipse or an exactly universal clock transformation.
                target = self.at(peak)[0][method] - 12
                all_x = sorted(points)
                for direction in (-1, 1):
                    candidates = sorted((x for x in all_x if direction*(x-peak) > 0),
                                        key=lambda x: abs(x-peak))
                    prior = peak
                    for x in candidates:
                        if self.at(x)[0][method] <= target:
                            root = brentq(lambda z: max(self.at(z)[0][method], -1e100) - target,
                                          min(prior, x), max(prior, x), xtol=1e-7)
                            boundaries.add(root)
                            points.add(root)
                            break
                        prior = x
        # The data-dependent diagnostic uses the solver likelihood under every
        # posterior, so its level-set boundaries are shared by all methods.
        xs = sorted(points)
        for a, b in zip(xs[:-1], xs[1:]):
            fa = self.at(a)[0][self.reference] - self.truth_loglike
            fb = self.at(b)[0][self.reference] - self.truth_loglike
            if fa * fb < 0:
                root = brentq(lambda x: max(self.at(x)[0][self.reference], -1e100) - self.truth_loglike,
                              a, b, xtol=1e-7)
                boundaries.add(root)
        cuts = np.array(sorted(boundaries))
        # Remove numerical duplicates at transitions without removing a finite
        # interval of meaningful size; endpoints retain the full prior bounds.
        cuts = cuts[np.r_[True, np.diff(cuts) > 1e-12]]
        self.discovery = dict(support_roots=support_roots, modes=modes, cuts=cuts.tolist())
        return np.column_stack([cuts[:-1], cuts[1:]])

    def panel(self, a, b):
        estimates = {}
        mixed = False
        for n, (z, w) in self.rules.items():
            x = (a+b)/2 + (b-a)/2*z
            weights = (b-a)/2*w
            lp = np.stack([self.at(v)[0] for v in x])
            likelihood = np.exp(lp)
            u = (x-self.lo)/(self.hi-self.lo)
            likelihood_cdf = lp[:, self.reference] < self.truth_loglike
            mixed |= bool(likelihood_cdf.any() and not likelihood_cdf.all())
            quantities = np.stack([np.ones(n), u, u*u, (x < self.truth_x).astype(float),
                                   likelihood_cdf.astype(float)], axis=1)
            value = np.einsum('n,nm,nk->mk', weights, likelihood, quantities)
            estimates[n] = (value, x, weights)
        fine, x, weights = estimates[5]
        error = np.abs(fine - estimates[3][0])
        if mixed:
            # Unresolved indicator changes receive the full cell mass as an
            # uncertainty estimate, even if the two rules agree accidentally.
            error[:, 4] = np.maximum(error[:, 4], fine[:, 0])
        return dict(value=fine, error=error, x=x, weights=weights)

    def integrate(self, tolerance=.002, max_panels=256):
        intervals = self.discover()
        panels = [self.panel(a, b) for a, b in intervals]
        while True:
            value = sum(p['value'] for p in panels)
            error = sum(p['error'] for p in panels)
            scale = np.maximum(value[:, :1], 1e-300)
            relative = error / scale
            if relative.max() <= tolerance or len(panels) >= max_panels:
                break
            priorities = [float(np.max(p['error']/scale)) for p in panels]
            i = int(np.argmax(priorities))
            a, b = intervals[i]
            if b-a < 1e-10:
                break
            midpoint = (a+b)/2
            intervals = np.concatenate([intervals[:i], [[a, midpoint], [midpoint, b]], intervals[i+1:]])
            panels[i:i+1] = [self.panel(a, midpoint), self.panel(midpoint, b)]
        return dict(value=value, error=error, relative_error=relative, intervals=intervals,
                    x=np.concatenate([p['x'] for p in panels]),
                    weights=np.concatenate([p['weights'] for p in panels]),
                    converged=bool(relative.max() <= tolerance), discovery=self.discovery,
                    evaluations=len(self.cache))


def outer_panel(a, b, columns, bounds, truth_y):
    """Nested positive Simpson rules; the y-CDF is classified per interval.

Splitting intervals at truth_y avoids assigning artificial endpoint atoms to
that CDF. Inner absolute errors are propagated through positive outer weights.
"""
    y = a + (b-a)*np.arange(5)/4
    values = np.stack([columns[float(v)]['value'] for v in y])
    errors = np.stack([columns[float(v)]['error'] for v in y])
    u = (y-bounds[0])/(bounds[1]-bounds[0])
    all_values = np.concatenate([values, values[:, :, :1]*u[:, None, None],
        values[:, :, :1]*u[:, None, None]**2,
        values[:, :, :1]*float((a+b)/2 < truth_y)], axis=2)
    all_errors = np.concatenate([errors, errors[:, :, :1]*u[:, None, None],
        errors[:, :, :1]*u[:, None, None]**2,
        errors[:, :, :1]*float((a+b)/2 < truth_y)], axis=2)
    weights = (b-a)*np.array([1, 4, 2, 4, 1])/12
    coarse = (b-a)*np.einsum('n,nmk->mk', np.array([1, 4, 1])/6, all_values[::2])
    fine = np.einsum('n,nmk->mk', weights, all_values)
    return dict(value=fine, error=np.abs(fine-coarse)+np.einsum('n,nmk->mk', weights, all_errors),
                y=y, weights=weights)
