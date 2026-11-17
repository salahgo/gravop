"""The tabulated closure reproduces gravop.closure for alpha-forms (table accuracy)."""
import os
import sys

import numpy as np
import pytest

EXT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
for p in (os.path.join(EXT, 'learned_closures'), os.path.join(EXT, 'scripts'), os.path.join(ROOT, 'src')):
    sys.path.insert(0, p)


@pytest.mark.parametrize('a,tol', [(1.0, 1e-11), (0.5, 1e-6)])
def test_table_reproduces_alpha_closure(a, tol):
    import gclosure as G
    import gfamilies as F
    from gravop import closure
    kw = dict(sigma_m=1.2, C=0.8, n_shells=120, eps_t=3e-4, timestep_mode='implicit',
              rho_c_factor=1e3, kn_min=0.02, t_end=400.0)
    ref = closure.run_halo_g(alpha=a, **kw)
    gx, gf = F.alpha(a)
    new = G.run_halo_t(gx=gx, gf=gf, **kw)
    assert abs(new['t_final'] / ref['t_final'] - 1) < tol


def test_rational_family_contains_harmonic():
    import gfamilies as F
    assert np.abs(F.rational(1.0, 2.0)[1]).max() < 1e-12
