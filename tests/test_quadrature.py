"""Numerical integration checks with analytic references, independent of SIDM."""
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.special import erf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gravop.quadrature import DyadicMesh


@pytest.mark.verification
def test_positive_rule_preserves_area_and_bilinear_integrals_after_local_refinement():
    mesh=DyadicMesh(base=4,lattice=256);mesh.refine([0,5,11])
    nodes,ids=mesh.stencil();xy=nodes/mesh.lattice;weights=mesh.weights(ids,len(nodes))
    assert (weights>=0).all()
    np.testing.assert_allclose(weights.sum(),1,atol=1e-14)
    f=1+xy[:,0]+2*xy[:,1]+3*xy[:,0]*xy[:,1]
    np.testing.assert_allclose(weights@f,3.25,atol=1e-14)
    np.testing.assert_allclose(mesh.assess(f,ids)['relative_error'],0,atol=1e-14)


@pytest.mark.verification
def test_adaptive_rule_resolves_separated_narrow_modes_against_analytic_integral():
    mesh=DyadicMesh(base=16,lattice=4096);sigma=.025
    for _ in range(80):
        nodes,ids=mesh.stencil();xy=nodes/mesh.lattice
        f=sum(np.exp(-np.square(xy-center).sum(1)/(2*sigma*sigma)) for center in ([.22,.31],[.73,.82]))
        r=mesh.assess(f,ids)
        if r['relative_error'].max()<.004:break
        mesh.refine(np.argsort(r['priority'])[-16:])
    exact=sum(np.prod([sigma*np.sqrt(np.pi/2)*(erf((1-c)/(np.sqrt(2)*sigma))+erf(c/(np.sqrt(2)*sigma))) for c in center]) for center in ([.22,.31],[.73,.82]))
    assert r['relative_error'].max()<.004
    assert abs(r['integral'][0]/exact-1)<.002


@pytest.mark.verification
def test_invalid_or_empty_integrands_do_not_report_convergence():
    mesh=DyadicMesh(base=2,lattice=32);nodes,ids=mesh.stencil()
    for v in (np.zeros(len(nodes)),np.full(len(nodes),np.nan),-np.ones(len(nodes))):
        with pytest.raises(ValueError):mesh.assess(v,ids)
