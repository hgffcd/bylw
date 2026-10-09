import numpy as np
from experiments.c146_loss_geometry.geometry import disk_margin,gap_and_dist
from experiments.c146_loss_geometry.belief import Belief,extrapolate
from experiments.c146_loss_geometry.controllers import Method
from experiments.c146_loss_geometry.world import Scenario,LossWorld


def triangle(r):
    a=np.arange(3)*2*np.pi/3
    return r*np.stack([np.cos(a),np.sin(a)],-1)


def test_exact_triangle_disk_certificate_against_dense_geometry():
    rng=np.random.default_rng(146)
    p=np.repeat(triangle(4.2)[None],80,0)+rng.normal(0,.4,(80,3,2))
    center=rng.normal(0,.2,(80,2))
    margin,_=disk_margin(p,center)
    assert (margin>0).all()
    a=np.linspace(0,2*np.pi,8192,endpoint=False)
    circle=np.stack([np.cos(a),np.sin(a)],-1)
    for scale,expected in [(.999,True),(1.001,False)]:
        q=center[:,None]+margin[:,None,None]*scale*circle
        gap,d=gap_and_dist(p[:,None],q)
        all_valid=((d.min(-1)>=1.5)&(d.max(-1)<=5.5)&(gap<=np.deg2rad(150))).all(-1)
        assert (all_valid==expected).all()


def test_regular_triangle_and_outside_target():
    for r in [3.5,4.2]:
        m,_=disk_margin(triangle(r),np.zeros(2))
        assert np.isclose(m,min(r-1.5,5.5-r,(2-np.sqrt(3))*r))
    m,_=disk_margin(triangle(4.2),np.array([9.,0.]))
    assert m<0


def test_hidden_truth_cannot_change_belief_or_action():
    seeds=list(range(1460100,1460108));m=Method(kind='disk_preview')
    a=LossWorld(seeds,Scenario(loss_start=0.1,loss_duration=2.),m)
    b=LossWorld(seeds,Scenario(loss_start=0.1,loss_duration=2.),m)
    a.observe();b.observe();a.time=b.time=.2
    b.target+=np.array([1000.,-123.]);b.tv*=400.
    oa=a.observe();ob=b.observe()
    for key in oa:assert np.array_equal(oa[key],ob[key])
    assert np.array_equal(a.nominal(oa),b.nominal(ob))


def test_constant_turn_integral_and_mask():
    p=np.array([[0.,0.]]);v=np.array([[2.,0.]])
    q,w=extrapolate(p,v,np.array([1.]),np.array([np.pi/2]),'ct')
    assert np.allclose(q,[[2,2]]) and np.allclose(w,[[0,2]])
    b=Belief(1,'ct');b.update(0,np.array([True]),p,v)
    b.update(.1,np.array([False]),np.full((1,2),np.nan),np.full((1,2),np.nan))
    assert np.allclose(b.query(.5)[0],[[1,0]])
