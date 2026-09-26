"""Independent checks for the repaired conditional/outer integration path."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from gravop.conditional_refinement import RefinedConditional, OuterMesh, feature_brackets, SolveBudget


@pytest.mark.verification
def test_verified_inner_partition_reuses_nodes_and_tightens_correctly():
    cache, batches = {}, []
    def value(x):
        return np.array([-.5*((x-.423)/.003)**2]), np.array([.8-x])
    def evaluate(x):
        if x not in cache:
            lp, support = value(x)
            cache[x] = (np.where(support >= 0, lp, -np.inf), support)
        return cache[x]
    def prefetch(xs):
        missing = [float(x) for x in xs if float(x) not in cache]
        batches.append(len(missing))
        for x in missing:
            evaluate(x)
    first = RefinedConditional(evaluate, (0, 1), .424, -.5, prefetch=prefetch).integrate(.002)
    count = len(cache)
    warm = {k: first[k] for k in ('intervals', 'discovery')}
    same = RefinedConditional(evaluate, (0, 1), .424, -.5, prefetch=prefetch, warm=warm).integrate(.002)
    assert len(cache) == count
    np.testing.assert_array_equal(first['value'], same['value'])
    fine = RefinedConditional(evaluate, (0, 1), .424, -.5, prefetch=prefetch, warm=warm).integrate(.000125)
    assert fine['converged'] and max(batches) >= 33
    np.testing.assert_allclose(fine['value'][0, 0], .003*np.sqrt(2*np.pi), rtol=1e-5)
    np.testing.assert_allclose(fine['value'][0, 3]/fine['value'][0, 0], ndtr(1/3), atol=2e-5)
    bad = dict(intervals=np.array([[0., .4], [.5, 1.]]), discovery={})
    with pytest.raises(ValueError):
        RefinedConditional(evaluate, (0, 1), .424, -.5, warm=bad).integrate()


@pytest.mark.verification
def test_outer_nodes_reuse_exactly_and_propagate_positive_errors():
    boundary = .6492923469631748
    mesh = OuterMesh([[0, boundary], [boundary, 1.]])
    old = set(mesh.required())
    mesh.refine([0, 1])
    assert old <= set(mesh.required())
    columns = {y: dict(value=np.ones((2, 5)), error=np.full((2, 5), 1e-5)) for y in mesh.required()}
    panels = mesh.assess(columns, boundary)
    total = sum(p['value'] for p in panels)
    np.testing.assert_allclose(total[:, [0, 5, 6, 7]], np.tile([1, .5, 1/3, boundary], (2, 1)), atol=1e-15)
    np.testing.assert_allclose(sum(p['inner_error'] for p in panels)[:, 0], 1e-5)
    assert all((p['weights'] > 0).all() for p in panels)
    with pytest.raises(ValueError):
        mesh.assess(columns, .12345)


def gaussian_column(y):
    """Exact truncated-normal moments/CDFs for two distinct curved ridges."""
    lo, hi, truth = -1.5, 1., -.29722113161818897
    centers = np.array([-.33+1.33*np.exp(-2*(y+.56)), -.328+1.33*np.exp(-2*(y+.56))])
    scales = np.array([.003, .004])
    peak = -.2+.2*np.tanh(2*(y-.1))
    a, b = (lo-centers)/scales, (hi-centers)/scales
    phi = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
    pa, pb = phi(a), phi(b)
    mass = ndtr(b)-ndtr(a)
    first = centers*mass+scales*(pa-pb)
    second = (centers**2+scales**2)*mass+2*centers*scales*(pa-pb)+scales**2*(a*pa-b*pb)
    cdf_x = np.maximum(0, ndtr((truth-centers)/scales)-ndtr(a))
    cdf_lp = mass.copy()
    if peak > -.2:
        radius = scales[0]*np.sqrt(2*(peak+.2))
        left, right = max(lo, centers[0]-radius), min(hi, centers[0]+radius)
        if left < right:
            cdf_lp -= ndtr((right-centers)/scales)-ndtr((left-centers)/scales)
    factor = np.exp(peak)*scales*np.sqrt(2*np.pi)
    value = np.stack([mass, (first-lo*mass)/(hi-lo),
        (second-2*lo*first+lo*lo*mass)/(hi-lo)**2, cdf_x, cdf_lp], axis=1)*factor[:, None]
    return dict(value=value, error=np.zeros((2, 5)))


@pytest.mark.verification
def test_complete_outer_method_resolves_curved_ridges_and_all_cdfs():
    truth = .6492923469631748
    mesh = OuterMesh([[-1., .25], [.25, truth], [truth, 1.5]])
    columns = {y: gaussian_column(y) for y in mesh.required()}
    mesh.localize(feature_brackets(columns))
    for tolerance in (.02, .005):
        while True:
            for y in mesh.required():
                if y not in columns:
                    columns[y] = gaussian_column(y)
            panels = mesh.assess(columns, truth)
            value = sum(p['value'] for p in panels)
            error = sum(p['error'] for p in panels)
            if np.max(error/value[:, :1]) <= tolerance:
                break
            assert len(columns) < 1000
            mesh.refine([int(np.argmax([np.max(p['error']/value[:, :1]) for p in panels]))])
        def integrand(y, method, component):
            v = gaussian_column(y)['value'][method]
            u = (y+1)/2.5
            return np.r_[v, v[0]*u, v[0]*u*u, v[0]*float(y < truth)][component]
        reference = np.array([[quad(integrand, -1., 1.5, args=(m, j),
            points=[-.57, -.56, -.55, .1, .6, truth, .7], epsabs=1e-11,
            epsrel=1e-10, limit=300)[0] for j in range(8)] for m in range(2)])
        assert np.max(abs(value-reference)/reference[:, :1]) <= tolerance
        mesh.refine(range(len(mesh.cells)))


@pytest.mark.verification
def test_solver_reservations_survive_restart_and_concurrency(tmp_path):
    path = tmp_path/'budget.json'
    budget = SolveBudget(path, 13, 13)
    def attempt(_):
        try:
            budget.reserve('column', 1)
            return True
        except RuntimeError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(40))) == 13
    assert SolveBudget(path, 13, 13).used == 13
    with pytest.raises(RuntimeError):
        budget.reserve('another', 1)
    with pytest.raises(ValueError):
        SolveBudget(path, 14, 13)
    separate = SolveBudget(tmp_path/'other.json', 10, 2)
    separate.reserve('a', 2)
    with pytest.raises(RuntimeError):
        separate.reserve('a', 1)
    separate.reserve('b', 2)
    assert separate.used == 4
