import numpy as np
import pytest
from experiments.c145_feasibility_reset.planar import Config,unit
from experiments.c146_loss_geometry.controllers import polar
from experiments.c210_radial_damping.controller import damping_delta,damped_polar


def observation(radius=7.,n=3):
    angles=np.arange(n)*2*np.pi/n
    p=radius*np.stack([np.cos(angles),np.sin(angles)],-1)[None]
    return dict(p=p,target=np.zeros((1,2)),v=unit(p)*2+np.array([1.1,.3]),
                target_v=np.array([[1.1,.3]]),measured=np.ones(1,bool),searcher=np.full(1,-1,int))


@pytest.mark.parametrize('n',[3,4,5])
def test_radial_sign_and_no_nominal_tangent(n):
    o=observation(n=n);er=unit(o['p']);et=np.stack([-er[...,1],er[...,0]],-1)
    d=damping_delta(o,Config(),np.array([.25]))
    np.testing.assert_allclose(np.sum(d*er,axis=-1),-.45,atol=1e-14)
    np.testing.assert_allclose(np.sum(d*et,axis=-1),0,atol=1e-14)
    o['v']=o['target_v'][:,None]-2*er
    np.testing.assert_allclose(damping_delta(o,Config(),[.25]),-d,atol=1e-14)


@pytest.mark.parametrize('disabled',['far','hidden','search','zero_relative_radial','zero_anticipation'])
def test_disabled_field_exactly_matches_original(disabled):
    o=observation(13 if disabled=='far' else 7,n=4);delay=np.array([.25])
    if disabled=='hidden':o['measured'][:]=False
    if disabled=='search':o['searcher'][:]=0
    if disabled=='zero_relative_radial':o['v']=np.broadcast_to(o['target_v'][:,None],o['p'].shape).copy()
    if disabled=='zero_anticipation':delay[:]=0
    original=polar(o['p'],o['v'],o['target'],o['target_v'],np.array([6.]),Config())
    np.testing.assert_array_equal(damped_polar(o,np.array([6.]),Config(),delay),original)


@pytest.mark.parametrize('n',[3,4,5])
def test_rigid_motion_permutation_common_velocity_and_nominal_dissipation(n):
    rng=np.random.default_rng(n);o=observation(n=n);o['v']+=rng.normal(size=o['v'].shape)
    d=damping_delta(o,Config(),[.25]);rel=o['v']-o['target_v'][:,None]
    assert np.sum(d*rel)<=1e-12
    a=2.4;R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    order=rng.permutation(n);transformed=dict(o,p=o['p'][:,order]@R+17,target=o['target']@R+17,
        v=o['v'][:,order]@R+4,target_v=o['target_v']@R+4)
    np.testing.assert_allclose(damping_delta(transformed,Config(),[.25]),d[:,order]@R,atol=1e-13)
    u=damped_polar(o,np.array([6.]),Config(),[.25]);assert (np.linalg.norm(u,axis=-1)<=4+1e-12).all()


def test_batched_transition_and_degenerate_finite():
    oo=[observation(r) for r in [7.,10.,13.]]
    o={k:np.concatenate([v[k] for v in oo]) for k in oo[0]}
    d=damping_delta(o,Config(),[.25]*3)
    np.testing.assert_allclose(np.linalg.norm(d,axis=-1)[:,0],[.45,.225,0],atol=1e-14)
    o['p'][0,0]=o['target'][0]
    assert np.isfinite(damped_polar(o,np.array([6.]*3),Config(),[.25]*3)).all()
