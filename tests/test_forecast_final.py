"""Observation-specific F8 contracts, independent of expensive solver runs."""
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
import pytest
import forecast_final_case as f8


@pytest.mark.verification
@pytest.mark.parametrize('j,t', [(0,100.), (1,400.), (2,900.)])
def test_case_uses_observed_physical_time_and_support(j, t, monkeypatch):
    calls = []
    class Ensemble:
        def predict(self, profile, params, time):
            calls.append(time)
            return np.full((1,1,20), 2.), np.array([.5])
    monkeypatch.setattr(f8, 'interpolate', lambda fields, radii: fields)
    obj = object.__new__(f8.Column)
    obj.y = .25
    obj.context = SimpleNamespace(time_index=j, time=t, ensemble=Ensemble(),
        profile=np.zeros((1,1,20)), obs={'observations':np.full((1,20), j+1.)})
    obj.rows = {0.: (np.stack([np.full(20,k+1.) for k in range(3)]),
                         np.array([True,True,False]), 1, 500.)}
    obj.predictions = {}
    obj.prefetch = lambda xs: None
    lp, support = obj.evaluate(0.)
    assert calls == [t]
    assert lp[0] == (0. if j < 2 else -np.inf)
    assert (support[0] > 0) == (j < 2)
    assert lp[1] == pytest.approx(-.5*20*((2-(j+1))/f8.NOISE_DEX)**2)


@pytest.mark.verification
def test_new_case_reuses_physics_without_inheriting_likelihood_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(f8, 'DATA', tmp_path)
    rows = {.1:(np.ones((3,20)), np.ones(3,bool), 0, 900.)}
    ctx = SimpleNamespace(seed_rows={.25:rows})
    col = f8.Column(.25,ctx)
    assert col.new_count == 0 and set(col.rows) == {.1}
    assert col.predictions == {}
    col.save()
    assert (col.path/'nodes.npz').exists()
    assert not list(col.path.glob('stage*'))
