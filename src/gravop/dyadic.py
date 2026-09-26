"""Outer nodes anchored to a fixed base interval for exact cache reuse.

Keep ``bounds`` unchanged when refining: represent descendants by their integer
``index`` and ``depth``. This avoids recomputing inherited nodes from rounded
child endpoints. The frozen F6 experiment does not import this revision.
"""
import operator

import numpy as np


def dyadic_nodes(bounds, index=0, depth=0):
    """Five nested Simpson nodes for one descendant of a fixed base interval.

    Children of (index, depth) are (2*index, depth+1) and
    (2*index+1, depth+1). Corresponding inherited coordinates are bit identical.
    Raise if the requested spacing cannot be represented distinctly.
    """
    index, depth = operator.index(index), operator.index(depth)
    a, b = map(float, bounds)
    if not np.isfinite([a, b]).all() or not a < b or not np.isfinite(b-a):
        raise ValueError('base bounds must be finite and increasing with finite span')
    if not 0 <= depth <= 48 or not 0 <= index < (1 << depth):
        raise ValueError('invalid dyadic panel index or depth')
    numerator = 4*index + np.arange(5, dtype=np.int64)
    denominator = 4 << depth
    fraction = numerator.astype(float)/denominator
    nodes = a+(b-a)*fraction
    nodes[numerator == 0] = a
    nodes[numerator == denominator] = b
    if not (np.diff(nodes) > 0).all():
        raise ValueError('dyadic node spacing is below floating-point resolution')
    return nodes
