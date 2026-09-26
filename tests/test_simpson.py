"""Analytic integration checks independent of the SIDM forward solver."""
import numpy as np
import pytest
from scipy.special import erf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gravop.simpson import SimpsonMesh


@pytest.mark.verification
def test_local_refinement_integrates_tensor_cubic_exactly():
    mesh = SimpsonMesh(base=4, lattice=256)
    mesh.refine([0, 5, 11])
    nodes, ids = mesh.stencil()
    xy = nodes / mesh.lattice
    weights = mesh.weights(ids, len(nodes))
    values = (1 + xy[:, 0]**3) * (1 + 2 * xy[:, 1]**3)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-14)
    np.testing.assert_allclose(weights @ values, 1.25 * 1.5, atol=1e-14)
    np.testing.assert_allclose(mesh.assess(values, ids)['relative_error'], 0, atol=1e-14)


@pytest.mark.verification
def test_fourth_order_convergence_and_embedded_difference():
    errors = []
    for base in (2, 4, 8):
        mesh = SimpsonMesh(base=base, lattice=256)
        nodes, ids = mesh.stencil()
        xy = nodes / mesh.lattice
        assessment = mesh.assess(xy[:, 0]**4 + xy[:, 1]**4, ids)
        actual = assessment['integral'][0] - .4
        errors.append(actual)
        np.testing.assert_allclose(assessment['relative_error'][0] * assessment['integral'][0],
                                   15 * actual, rtol=1e-8)
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 16, rtol=1e-8)


@pytest.mark.verification
def test_adaptive_thin_diagonal_ridge_against_analytic_integral():
    mesh = SimpsonMesh(base=16, lattice=16384)
    sigma = .008
    for _ in range(100):
        nodes, ids = mesh.stencil()
        xy = nodes / mesh.lattice
        values = np.exp(-.5 * ((xy[:, 0] - xy[:, 1]) / sigma)**2)
        assessment = mesh.assess(values, ids)
        if assessment['relative_error'][0] <= .005:
            break
        mesh.refine(np.argsort(assessment['priority'])[-16:])
    exact = 2 * (sigma * np.sqrt(np.pi / 2) * erf(1 / (np.sqrt(2) * sigma))
                 - sigma**2 * (1 - np.exp(-.5 / sigma**2)))
    assert assessment['relative_error'][0] <= .005
    assert abs(assessment['integral'][0] / exact - 1) < .001


@pytest.mark.verification
def test_minimum_spacing_and_invalid_mass_fail_explicitly():
    with pytest.raises(ValueError):
        SimpsonMesh(base=4, lattice=8)
    mesh = SimpsonMesh(base=4, lattice=16)
    nodes, ids = mesh.stencil()
    with pytest.raises(ValueError):
        mesh.refine([0])
    for values in (np.zeros(len(nodes)), np.full(len(nodes), np.nan), -np.ones(len(nodes))):
        with pytest.raises(ValueError):
            mesh.assess(values, ids)
