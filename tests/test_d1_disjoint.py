import pytest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from statecond.pdes import SYSTEMS  # noqa: E402
_BEFORE = dict(SYSTEMS)
import d1_systems  # noqa: E402,F401
import d1_campaign as dc  # noqa: E402
NEW = {n: SYSTEMS[n] for n in dc.ORDER}
SYSTEMS.clear(); SYSTEMS.update(_BEFORE)  # keep the suite's parametrised tests on the original systems


@pytest.mark.verification
def test_new_systems_registered_and_singular():
    for name in dc.ORDER:
        S = NEW[name]()
        tr = S.solve(S.sample(np.random.default_rng(557_000_000)))
        assert tr is not None and tr.u.shape[1:] == (S.n_fields, 128) and np.isfinite(tr.u).all()
        if name != 'bimodal':
            assert tr.status == 0 and tr.a[-1] >= 0.99 * S.blowup_factor * tr.a[0]


@pytest.mark.verification
def test_seed_blocks_disjoint_from_prototypes_and_suite():
    prod = {dc.SEED0 + 100_000 * k + i for k in range(6) for i in range(600)}
    proto = set(range(555_000_000, 555_000_064)) | set(range(556_000_000, 556_000_008)) | {557_000_000}
    suite = {100_000 * k + i for k in range(5) for i in range(600)}
    assert not prod & proto and not prod & suite


@pytest.mark.verification
def test_aggregate_identity_and_bootstrap_resample():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(3, 20, 50)); Gv = G + 0.3 * rng.normal(size=G.shape); Gh = Gv + 0.2 * rng.normal(size=G.shape); Ga = G + 0.2 * rng.normal(size=G.shape)
    S = dict(D2=((Gv - G) ** 2).sum(2), Ed2=((Gh - Gv) ** 2).sum(2), EA2=((Ga - G) ** 2).sum(2), Dep2=((Gh - G) ** 2).sum(2),
             X=((Gv - G) * (Gh - Gv)).sum(2), n_el=50)
    agg = dc.aggregate(dc.per_seed_from_sums(S))
    assert abs(agg['closure'] - 1) < 1e-12
    r = dc.residuals(agg, 0.1)
    assert abs(r['r_identity'] - r['r_quadrature']) < 1e-12
    full = dc.per_seed_from_sums(S, np.arange(20)); base = dc.per_seed_from_sums(S)
    assert all(abs(x['F'] - y['F']) < 1e-14 for x, y in zip(full, base))
    rt = dc.per_trajectory_residuals(S); assert rt.shape == (20,) and np.isfinite(rt).all()
