"""Integration safeguards that matter independently of the selected observations."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from forecast_numerics_pilot import compare, seeded_mesh


@pytest.mark.verification
def test_seeding_irregular_modes_preserves_full_prior_and_samples_each_seed():
    selection = dict(seed_keys=[[16, 65520], [32208, 48688], [0, 0], [65536, 65536]])
    mesh = seeded_mesh(selection)
    nodes, indices = mesh.stencil()
    identities = {tuple(k) for k in nodes}
    assert all(tuple(k) in identities for k in selection['seed_keys'])
    weights = mesh.weights(indices, len(nodes))
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-14)
    np.testing.assert_allclose(weights @ (nodes / mesh.lattice), [.5, .5], atol=1e-14)


@pytest.mark.verification
def test_stable_moments_do_not_override_unstable_likelihood_cdf():
    first = dict(mean=np.zeros((2, 2)), sd=np.ones((2, 2)),
                 width90=np.ones((2, 2)), cdf=np.full((2, 3), .5))
    second = {k: v.copy() for k, v in first.items()}
    assert compare(first, second)['passed']
    second['cdf'][1, 2] += .03
    assert not compare(first, second)['passed']
