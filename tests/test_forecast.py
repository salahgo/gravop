"""Deployment contracts for the physical-time forecasting model."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from statecond.forecast import ClockConditionedOperator, ForecastPredictor
from sc_experiment import field_err

@pytest.mark.verification
@pytest.mark.parametrize('channels', [1,2])
def test_initial_condition_and_frozen_clock(channels):
    model = ClockConditionedOperator(channels, 3, n_x=31, width=8, n_layers=2)
    prof, par = torch.randn(3,channels,31), torch.randn(3,3)
    out = model(prof,par,torch.tensor([0.,.1,1.]))
    assert torch.equal(out[0],prof[0])
    out.square().mean().backward()
    assert all(p.grad is None for p in model.clock.parameters())
    assert any(p.grad is not None for p in model.blocks.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.verification
def test_field_evaluation_disables_dropout_and_batchnorm_updates():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn=torch.nn.BatchNorm1d(1)
            self.drop=torch.nn.Dropout(.5)
        def forward(self,p,c): return self.drop(self.bn(p))
    net=Net().train()
    u=torch.randn(3,2,1,16); p=u[:,0]
    before=net.bn.running_mean.clone()
    cond=lambda ri,ti: torch.zeros(len(ri),1)
    a=field_err(net,cond,u,p,[0,1,2]); b=field_err(net,cond,u,p,[0,1,2])
    np.testing.assert_array_equal(a,b)
    assert torch.equal(net.bn.running_mean,before)


@pytest.mark.verification
def test_standalone_checkpoint_and_input_validation(tmp_path):
    cfg=dict(n_fields=1,n_par=2,n_x=16,width=8,n_layers=1)
    model=ClockConditionedOperator(**cfg).eval()
    norm=dict(f_mu=np.array([[[[2.]]]]),f_sd=np.array([[[[3.]]]]),
              p_mu=np.zeros((1,2)),p_sd=np.ones((1,2)))
    path=tmp_path/'model.pt'
    torch.save(dict(config={'model':cfg},state=model.state_dict(),norm=norm),path)
    pred=ForecastPredictor(path)
    prof=np.random.default_rng(0).normal(2.,3.,(2,1,16)).astype(np.float32)
    par=np.zeros((2,2))
    out=pred.predict(prof,par,0.)
    np.testing.assert_array_equal(out['fields'],prof)
    assert out['within_predicted_horizon'].all()
    with pytest.raises(ValueError,match='nonnegative'): pred.predict(prof,par,-1.)
    with pytest.raises(ValueError,match='spatial points'): pred.predict(prof[...,:8],par,1.)


@pytest.mark.verification
def test_invalid_checkpoint_scales_and_overflowing_clock_fail_explicitly(tmp_path):
    cfg=dict(n_fields=1,n_par=2,n_x=16,width=8,n_layers=1)
    model=ClockConditionedOperator(**cfg).eval()
    norm=dict(f_mu=np.zeros(1),f_sd=np.ones(1),p_mu=np.zeros(2),p_sd=np.ones(2))
    path=tmp_path/'model.pt'
    torch.save(dict(config={'model':cfg},state=model.state_dict(),norm=norm),path)
    predictor=ForecastPredictor(path)
    class OverflowClock(torch.nn.Module):
        def forward(self,p,c): return torch.full((len(p),),100.)
    predictor.model.clock=OverflowClock()
    with pytest.raises(FloatingPointError,match='terminal time'):
        predictor.predict(np.zeros((1,1,16)),np.zeros((1,2)),1.)
    norm['p_sd'][0]=0
    torch.save(dict(config={'model':cfg},state=model.state_dict(),norm=norm),path)
    with pytest.raises(ValueError,match='positive scales'): ForecastPredictor(path)
