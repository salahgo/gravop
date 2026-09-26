"""Known integrals for conditional peaks, survival and diagnostic CDF cuts."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gravop.conditional import ConditionalIntegral, outer_panel


@pytest.mark.verification
def test_narrow_off_grid_peak_and_likelihood_cdf_against_normal_integrals():
    center, sigma, cut = .423, .002, .424
    def evaluate(x):
        lp = -.5*((x-center)/sigma)**2 if x < .8 else -np.inf
        return [lp], [.8-x]
    result = ConditionalIntegral(evaluate, (0, 1), cut, -.5).integrate(.001)
    assert result['converged']
    value = result['value'][0]
    np.testing.assert_allclose(value[0], sigma*np.sqrt(2*np.pi), rtol=1e-4)
    np.testing.assert_allclose(value[1]/value[0], center, atol=1e-6)
    np.testing.assert_allclose(value[3:]/value[0], [ndtr(.5), 2*ndtr(-1)], atol=2e-4)
    assert (result['weights'] > 0).all()
    np.testing.assert_allclose(result['weights'].sum(), 1, atol=1e-12)


@pytest.mark.verification
def test_disconnected_support_and_multiple_modes_remain_in_the_integral():
    def evaluate(x):
        metric = max(min(x, .4-x), min(x-.6, 1-x))
        lp = np.logaddexp(-.5*((x-.2)/.01)**2, -.5*((x-.8)/.01)**2)
        return [lp if metric >= 0 else -np.inf], [metric]
    result = ConditionalIntegral(evaluate, (0, 1), .5, -1).integrate(.0001)
    assert result['converged']
    value = result['value'][0]
    np.testing.assert_allclose(value[0], 2*.01*np.sqrt(2*np.pi), rtol=1e-4)
    np.testing.assert_allclose(value[1]/value[0], .5, atol=1e-6)
    np.testing.assert_allclose(value[3]/value[0], .5, atol=1e-6)
    assert len(result['discovery']['modes']) >= 2


@pytest.mark.verification
def test_outer_cdf_uses_panel_sides_and_propagates_inner_uncertainty():
    cut = .37
    panels = []
    for a, b in ((-1, cut), (cut, 1.5)):
        ys = a+(b-a)*np.arange(5)/4
        columns = {float(y): dict(value=np.array([[1., .5, 1/3, .5, .2]]),
                                 error=np.full((1, 5), .001)) for y in ys}
        panels.append(outer_panel(a, b, columns, (-1, 1.5), cut))
    value = sum(p['value'] for p in panels)[0]
    error = sum(p['error'] for p in panels)[0]
    np.testing.assert_allclose(value[0], 2.5, atol=1e-14)
    np.testing.assert_allclose(value[7]/value[0], (cut+1)/2.5, atol=1e-14)
    assert error[0]/value[0] >= .001-1e-14


@pytest.mark.verification
def test_inconsistent_support_and_nonfinite_inputs_fail():
    for callback in (lambda x: ([0.], [-1.]), lambda x: ([np.nan], [1.])):
        with pytest.raises(ValueError):
            ConditionalIntegral(callback, (0, 1), .5, -1).integrate()


@pytest.mark.verification
def test_candidate_mass_outside_solver_support_is_retained_in_likelihood_cdf():
    def evaluate(x):
        support = np.array([.4-x, .8-x])
        return np.where(support >= 0, 0., -np.inf), support
    result = ConditionalIntegral(evaluate, (0, 1), .3, -1).integrate(.001)
    assert result['converged']
    values = result['value']
    np.testing.assert_allclose(values[:, 0], [.4, .8], atol=1e-6)
    np.testing.assert_allclose(values[:, 3]/values[:, 0], [.75, .375], atol=1e-6)
    np.testing.assert_allclose(values[:, 4]/values[:, 0], [0., .5], atol=1e-6)
