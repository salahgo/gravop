"""Independent probability/likelihood contracts for F3."""
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import multivariate_normal, kstest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from gravop.calibration import (GaussianDiscrepancy, posterior_weights,
                                randomized_pit, select_discrepancy, holm)


@pytest.mark.verification
def test_correlated_likelihood_and_bias_match_scipy():
    bias=np.array([.2,-.3]); covariance=np.array([[.2,.08],[.08,.1]])
    model=GaussianDiscrepancy(bias,covariance,.1)
    residual=np.random.default_rng(12).normal(size=(5,2))
    expected=multivariate_normal.logpdf(residual,mean=bias,cov=covariance+.01*np.eye(2))
    np.testing.assert_allclose(model.logpdf(residual),expected,rtol=1e-12)
    with pytest.raises(ValueError,match='semidefinite'):
        GaussianDiscrepancy(bias,np.array([[1,2],[2,1]]),.1)


@pytest.mark.verification
def test_log_weights_handle_large_scale_and_reject_empty_support():
    np.testing.assert_allclose(posterior_weights([-10000,-10001,-np.inf]),
                               [1/(1+np.exp(-1)),1/(1+np.exp(1)),0])
    for bad in ([-np.inf,-np.inf],[0,np.nan],[0,np.inf]):
        with pytest.raises(ValueError): posterior_weights(bad)
    weights=np.array([.3,.3,.4])*(1+1e-15)
    assert randomized_pit([0,1,2],weights,3,.5)==1.
    with pytest.raises(ValueError,match='bounds'):
        randomized_pit([0,1,2],np.array([.3,.3,.4])*(1+1e-7),3,.5)


@pytest.mark.verification
def test_randomized_ties_and_data_dependent_negative_control():
    # An exactly enumerated, informative three-state Bayesian model.
    rng=np.random.default_rng(57); theta=np.array([-2.,0.,2.])
    oracle=[]; ignoring_data=[]; ignoring_parameter=[]
    for _ in range(2500):
        truth=rng.integers(3); y=theta[truth]+rng.normal(0,.3)
        lp=-.5*((y-theta)/.3)**2; q=posterior_weights(lp); u=rng.random()
        oracle.append(randomized_pit(lp,q,lp[truth],u))
        ignoring_data.append(randomized_pit(lp,np.ones(3)/3,lp[truth],u))
        ignoring_parameter.append(randomized_pit(theta,np.ones(3)/3,theta[truth],u))
    assert kstest(oracle,'uniform').pvalue>.001
    assert kstest(ignoring_parameter,'uniform').pvalue>.001
    assert kstest(ignoring_data,'uniform').pvalue<1e-20
    assert randomized_pit([1,1,2],[.2,.3,.5],1,.4)==pytest.approx(.2)


@pytest.mark.verification
def test_discrepancy_selection_detects_independent_correlated_error():
    rng=np.random.default_rng(2)
    residual=rng.multivariate_normal([.4,-.2],[[.1,.08],[.08,.1]],size=1500)
    model,record=select_discrepancy(residual[:1000],residual[1000:],.02,rng)
    assert record['selected']=='bias_cov_0.1'
    np.testing.assert_allclose(model.bias,[.4,-.2],atol=.03)
    assert model.covariance[0,1]>.05


@pytest.mark.verification
def test_holm_controls_familywise_order():
    np.testing.assert_allclose(holm([.04,.001,.03,.5]),[.09,.004,.09,.5])
