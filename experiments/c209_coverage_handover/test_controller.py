import numpy as np
import pytest
from experiments.c209_coverage_handover.controller import coverage_fraction,team_weight


def observation(angles,radius=13.,measured=True):
    angles=np.deg2rad(angles)
    return dict(p=radius*np.stack([np.cos(angles),np.sin(angles)],-1)[None],
                target=np.zeros((1,2)),measured=np.array([measured]))


@pytest.mark.parametrize('angles,expected',[
    ([0,0,0],0.),([0,30,60],1/3),([170,-170,180],1/9),
    ([0,90,180],1.),([0,120,240],1.),([0,90,180,270],1.)])
def test_known_angular_geometries(angles,expected):
    o=observation(angles)
    np.testing.assert_allclose(coverage_fraction(o['p'],o['target']),expected,atol=1e-14)


@pytest.mark.parametrize('n',[3,4,5])
def test_rigid_motion_permutation_and_scale_invariance(n):
    rng=np.random.default_rng(209+n);p=rng.normal(size=(16,n,2));target=rng.normal(size=(16,2))
    angle=2.8;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    expected=coverage_fraction(p,target)
    actual=coverage_fraction(2.7*p[:,rng.permutation(n)]@R+17,2.7*target@R+17)
    np.testing.assert_allclose(actual,expected,atol=1e-13)


def test_semicircle_boundary_continuity_and_degenerate_bearing():
    values=[]
    for angle in [180-1e-6,180,180+1e-6]:
        o=observation([0,90,angle]);values.append(coverage_fraction(o['p'],o['target'])[0])
    assert max(values)-min(values)<1e-7
    o['p'][0,0]=o['target'][0]
    assert coverage_fraction(o['p'],o['target'])[0]==0


def test_hidden_near_and_batched_weight():
    examples=[observation([0,30,60]),observation([0,120,240],10),
              observation([0,120,240],7),observation([0,120,240],13,False)]
    o={k:np.concatenate([x[k] for x in examples]) for k in examples[0]}
    np.testing.assert_allclose(team_weight(o),[1/3,.5,0,0],atol=1e-14)
    # Unrelated private fields are never read by the public geometry function.
    o.update(hidden_route=np.full((100,2),np.nan),target_truth=np.full((4,2),np.nan))
    np.testing.assert_allclose(team_weight(o),[1/3,.5,0,0],atol=1e-14)
