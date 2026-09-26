"""Population and statistical contracts for the independent F2 validation."""
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from forecast_validate import from_d1, comparison, Reference


@pytest.mark.verification
def test_full_application_population_preserves_horizon_runs_and_row_identity(tmp_path):
    path=tmp_path/'d1.h5'
    status=np.array([0,1,2,-1,3,4])
    with h5py.File(path,'w') as f:
        for k in ('lrho','lv'): f[k]=np.ones((6,48,128),dtype='f4')
        for k in ('log_sigma0','log_w','model_id','log_tc','ic_type','seed'):
            f[k]=np.arange(6)
        f['status']=status
    full=from_d1(path); terminal=from_d1(path,True)
    np.testing.assert_array_equal(full['source_index'],[0,1,2])
    np.testing.assert_array_equal(full['status'],[0,1,2])
    np.testing.assert_array_equal(terminal['source_index'],[0,2])
    np.testing.assert_array_equal(terminal['seed'],[0,2])


@pytest.mark.verification
def test_paired_bootstrap_preserves_constant_error_ratio():
    rng=np.random.default_rng(17)
    candidate=rng.lognormal(size=(3,25))
    out=comparison(2*candidate,candidate)
    np.testing.assert_allclose(out['ci95'],[2.,2.],rtol=1e-12)
    np.testing.assert_allclose(out['seed_and_trajectory_ci95'],[2.,2.],rtol=1e-12)


@pytest.mark.verification
def test_deployed_reference_uses_predicted_clock_and_finite_direct_zero_time():
    class Clock(torch.nn.Module):
        def forward(self,p,c): return torch.ones(len(p))
    class Net(torch.nn.Module):
        def forward(self,p,c): return c[:,-1,None,None].expand_as(p)
    class Norm:
        t_mu=0.; t_sd=1.
    p=torch.zeros(2,1,8); c=torch.zeros(2,3)
    aligned=Reference(Net(),Clock(),Norm(),3.)
    np.testing.assert_allclose(aligned(p,c,torch.tensor([2.,20.])).numpy()[:,0,0],[.2,1.])
    direct=Reference(Net(),Clock(),Norm(),3.,True)
    np.testing.assert_allclose(direct(p,c,torch.tensor([0.,3.])).numpy()[:,0,0],[0.,np.log(2.)])
