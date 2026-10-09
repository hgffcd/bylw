import numpy as np
import pytest
from experiments.c209_coverage_handover.controller import team_weight as parent_weight
from experiments.c211_uniformity_handover.controller import uniformity_fraction,team_weight


def obs(angles,radius=13.,seen=True):
    angles=np.deg2rad(angles)
    return dict(p=radius*np.stack([np.cos(angles),np.sin(angles)],axis=-1)[None],
                target=np.zeros((1,2)),measured=np.array([seen]))


@pytest.mark.parametrize('angles,factor',[
    ([0,120,240],1.),([0,90,180],2/3),([0,90,210],.8),
    ([0,90,180,270],1.),([0,60,120,180],.5),([0,0,0],1/3),([170,-170,180],1/2.833333333333333)])
def test_known_gap_geometries(angles,factor):
    o=obs(angles)
    np.testing.assert_allclose(uniformity_fraction(o['p'],o['target']),factor,atol=1e-14)


@pytest.mark.parametrize('n',[3,4,5])
def test_rigid_motion_scale_and_permutation_invariance(n):
    rng=np.random.default_rng(211+n);p=rng.normal(size=(24,n,2));target=rng.normal(size=(24,2))
    theta=2.8;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    got=uniformity_fraction(2.7*p[:,rng.permutation(n)]@R+17,2.7*target@R+17)
    np.testing.assert_allclose(got,uniformity_fraction(p,target),atol=1e-13)
    o=dict(p=15*p,target=target,measured=np.ones(24,bool))
    assert (team_weight(o)>=0).all() and (team_weight(o)<=parent_weight(o)+1e-14).all()


def test_boundary_continuity_degeneracy_and_hidden_near_identity():
    values=[uniformity_fraction(obs([0,90,a])['p'],np.zeros((1,2)))[0] for a in [180-1e-6,180,180+1e-6]]
    assert max(values)-min(values)<1e-7
    for o in [obs([0,90,180],7),obs([0,90,180],13,False)]:
        np.testing.assert_array_equal(team_weight(o),parent_weight(o))
    o=obs([0,90,180]);o['p'][0,0]=o['target'][0]
    assert uniformity_fraction(o['p'],o['target'])[0]==team_weight(o)[0]==0


def test_uniform_and_nonuniform_teams_at_multiple_distances():
    examples=[obs([0,120,240],13),obs([0,90,180],13),obs([0,90,180],10),obs([0,90,180],7)]
    o={k:np.concatenate([x[k] for x in examples]) for k in examples[0]}
    np.testing.assert_allclose(team_weight(o),[1,2/3,1/3,0],atol=1e-14)
    o.update(hidden_route=np.full((20,2),np.nan),target_truth=np.full((4,2),np.nan))
    np.testing.assert_allclose(team_weight(o),[1,2/3,1/3,0],atol=1e-14)
