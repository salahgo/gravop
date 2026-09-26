"""Independent-data and finite-sample rank contracts for prospective confirmation."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from forecast_fresh_confirmation import prerequisite,single_draw_rank


@pytest.mark.verification
def test_fresh_generation_requires_all_prerequisites(tmp_path):
    import json
    with pytest.raises(RuntimeError,match='blocked'): prerequisite(tmp_path)
    folder=tmp_path/'results/forecast_final';folder.mkdir(parents=True)
    path=folder/'diagnostics_passed.json'
    d={'diagnostics_passed':True,'diagnostics':[{'passed':True}]*5+[{'passed':False}]}
    path.write_text(json.dumps(d))
    with pytest.raises(RuntimeError,match='blocked'): prerequisite(tmp_path)
    d['diagnostics'][-1]['passed']=True;path.write_text(json.dumps(d))
    assert prerequisite(tmp_path)==path


@pytest.mark.verification
def test_one_draw_rank_is_uniform_for_exchangeable_draws_and_randomized_ties():
    rng=np.random.default_rng(20260911)
    truth,draw=rng.normal(size=(2,20000));u=rng.uniform(size=20000)
    ranks=single_draw_rank(draw,truth,u)
    histogram=np.histogram(ranks,np.linspace(0,1,11))[0]/len(ranks)
    assert np.max(abs(histogram-.1))<.01
    np.testing.assert_array_equal(single_draw_rank(truth,truth,u),u)
    # A posterior always on one side of truth must not look uniform.
    bad=single_draw_rank(truth-1,truth,u)
    assert bad.min()>=.5


@pytest.mark.verification
def test_reporting_amendment_reports_the_failure_rate_instead_of_stopping():
    """The 2026-09-11 amendment: attempt every observation, report the rate.

    The frozen rule returned at the first required numerical failure, so its
    most likely outcome was no reportable quantity at all. These are the
    contracts the replacement has to keep.
    """
    from forecast_fresh_confirmation import numerical_summary
    completed=[{'case':i,'passed':True} for i in range(8)]
    clean=numerical_summary(completed)
    assert clean['attempted']==8 and clean['converged']==8
    assert clean['failed_cases']==[] and clean['failure_rate']==0
    # A perfect run still carries honest uncertainty: 8/8 does not prove 100%.
    assert clean['failure_rate_ci95'][1]>0

    completed[3]['passed']=False
    completed[6]={'case':6,'passed':False,'error':'CalledProcessError: 1'}
    mixed=numerical_summary(completed)
    assert mixed['attempted']==8 and mixed['converged']==6
    assert mixed['failed_cases']==[3,6]
    assert mixed['failure_rate']==pytest.approx(.25)
    lo,hi=mixed['failure_rate_ci95']
    assert 0<lo<.25<hi<1
    assert mixed['convergence_rate']==pytest.approx(.75)
    np.testing.assert_allclose(mixed['convergence_rate_ci95'],[1-hi,1-lo])

    with pytest.raises(ValueError): numerical_summary([])


@pytest.mark.verification
def test_the_strict_population_claim_still_requires_zero_numerical_failures():
    """The amendment must not silently upgrade the frozen claim.

    Screens computed on the converged subset are conditional on numerical
    success, which is a selection towards easier posteriors. The strict flag
    therefore still requires zero failures; only the conditional flag may pass.
    """
    import inspect
    from forecast_fresh_confirmation import summarize
    src=inspect.getsource(summarize)
    assert "population_calibration_passed=bool(screens and not failed)" in src
    assert 'population_calibration_passed_conditional' in src
    assert 'selection_caveat' in src
