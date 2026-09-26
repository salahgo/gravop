from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gravop.dyadic import dyadic_nodes

@pytest.mark.verification
def test_inherited_nodes_keep_exact_cache_keys():
    bounds = (0.6492923469631748, 1.5)
    parent = dyadic_nodes(bounds)
    a, b = bounds
    rounded_child = a+((a+b)/2-a)*np.arange(5)/4
    assert rounded_child[2] != parent[1]  # Reproduce the F6 cache miss.
    rng = np.random.default_rng(917)
    for depth in range(17):
        indices = {0, (1 << depth)-1, int(rng.integers(1 << depth))}
        for index in indices:
            parent = dyadic_nodes(bounds, index, depth)
            left = dyadic_nodes(bounds, 2*index, depth+1)
            right = dyadic_nodes(bounds, 2*index+1, depth+1)
            inherited = np.r_[left[::2], right[2::2]]
            assert inherited.tobytes() == parent.tobytes()


@pytest.mark.verification
def test_adjacent_base_panels_share_original_endpoint():
    boundary = 0.6492923469631748
    left = dyadic_nodes((0.25, boundary), 7, 3)
    right = dyadic_nodes((boundary, 1.5), 0, 3)
    assert left[-1].tobytes() == right[0].tobytes() == np.float64(boundary).tobytes()


@pytest.mark.verification
def test_invalid_or_unrepresentable_panels_fail():
    for bounds, index, depth in [((1, 0), 0, 0), ((0, np.inf), 0, 0),
                                  ((0, 1), 2, 1), ((0, 1), 0, -1),
                                  ((0, 1), 0, 49), ((1, np.nextafter(1., 2.)), 0, 0)]:
        with pytest.raises(ValueError):
            dyadic_nodes(bounds, index, depth)
    with pytest.raises(TypeError):
        dyadic_nodes((0, 1), .5, 1)
