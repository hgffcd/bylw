import numpy as np
import pytest
from experiments.c212_predictive_spread.controller import forecast, spread_filter


def case(n=3):
    a=np.arange(n)*2*np.pi/n; er=np.stack([np.cos(a), np.sin(a)], -1)[None]
    r=np.linspace(5.2, 6.8, n)[None, :, None]
    v=np.linspace(-2, 2, n)[None, :, None]*er
    return dict(p=r*er, target=np.zeros((1,2)), v=v, target_v=np.zeros((1,2)),
        measured=np.ones(1,bool), searcher=np.full(1,-1), own_acceleration=np.zeros_like(v),
        acceleration_valid=np.ones(1,bool))


@pytest.mark.parametrize('n',[3,4,5])
def test_predictive_spread_contracts_with_bounded_radial_change(n):
    o=case(n);base=np.zeros_like(o['p']);u,d=spread_filter(base,o,diagnostics=True)
    assert d['enabled'].all() and (d['after_spread']<d['before_spread']).all()
    assert d['correction_mps'].max()<=.6+1e-12
    assert np.linalg.norm(u,axis=-1).max()<=4+1e-12
    before,_,_=forecast(o['p'],o['v'],o['own_acceleration'],base)
    after,_,_=forecast(o['p'],o['v'],o['own_acceleration'],u)
    np.testing.assert_allclose(after,before+d['jacobian']*(u-base),atol=1e-12)
    assert np.ptp(np.linalg.norm(after,axis=-1),axis=-1).item() < d['before_spread'].item()


@pytest.mark.parametrize('disabled',['hidden','search','far','missing_history','uniform'])
def test_disabled_output_exactly_unchanged(disabled):
    o=case();u=np.zeros_like(o['p'])
    if disabled=='hidden':o['measured'][:]=False
    if disabled=='search':o['searcher'][:]=1
    if disabled=='far':o['p']*=3
    if disabled=='missing_history':o['acceleration_valid'][:]=False
    if disabled=='uniform':o['v'][:]=0
    np.testing.assert_array_equal(spread_filter(u,o),u)


def test_forecast_jacobian_and_rigid_symmetry():
    rng=np.random.default_rng(212);o=case(4);a=rng.normal(size=o['p'].shape);u=rng.normal(size=a.shape)
    p,_,j=forecast(o['p'],o['v'],a,u);q,_,_=forecast(o['p'],o['v'],a,u+.37)
    np.testing.assert_allclose(q,p+j*.37,atol=1e-12)
    rotation=np.array([[0.,-1.],[1.,0.]]);ids=[2,0,3,1]
    transformed=dict(o,p=o['p'][:,ids]@rotation+12,target=o['target']@rotation+12,
        v=o['v'][:,ids]@rotation, target_v=o['target_v']@rotation,
        own_acceleration=o['own_acceleration'][:,ids]@rotation)
    np.testing.assert_allclose(spread_filter(u[:,ids]@rotation,transformed),spread_filter(u,o)[:,ids]@rotation,atol=1e-12)


def test_batched_speed_feasibility_and_no_private_field_dependence():
    rng=np.random.default_rng(42);o=case(4)
    o={k:np.repeat(v,128,axis=0) for k,v in o.items()}
    o['v']+=rng.normal(size=o['v'].shape);o['own_acceleration']+=rng.normal(size=o['v'].shape)
    u=rng.normal(size=o['p'].shape);u*=4/np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),4)
    out,d=spread_filter(u,o,diagnostics=True)
    assert np.isfinite(out).all() and (np.linalg.norm(out,axis=-1)<=4+1e-12).all()
    assert (d['after_spread']<=d['before_spread']+1e-12).all()
    np.testing.assert_array_equal(spread_filter(u,dict(o,hidden_target=np.ones((128,2))*1e6,route=np.nan)),out)
