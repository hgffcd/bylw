from types import SimpleNamespace
import numpy as np
import pytest
from experiments.c200_feedforward_budget.controller import allocate,budget_polar
from experiments.c208_metric_budget.controller import metric_correction,metric_budget_polar,formation_potential

C=SimpleNamespace(radial_gain=.9,angular_gain=.65,pursuer_speed=4.)


@pytest.mark.parametrize('agents',[3,4,5])
def test_negative_gradient_by_independent_coordinate_finite_differences(agents):
    rng=np.random.default_rng(208+agents)
    angles=np.arange(agents)*2*np.pi/agents+rng.uniform(-.1,.1,(12,agents))
    radius=rng.uniform(4,20,(12,agents));target=rng.uniform(-10,10,(12,2))
    p=target[:,None]+radius[...,None]*np.stack([np.cos(angles),np.sin(angles)],-1)
    q=metric_correction(p,target,np.full(12,6.),C);eps=1e-5
    for i in range(agents):
        for j in range(2):
            plus=p.copy();minus=p.copy();plus[:,i,j]+=eps;minus[:,i,j]-=eps
            derivative=(formation_potential(plus,target,np.full(12,6.),C)-formation_potential(minus,target,np.full(12,6.),C))/(2*eps)
            np.testing.assert_allclose(q[:,i,j],-derivative,atol=1e-7,rtol=1e-7)


def test_ideal_velocity_budget_preserves_potential_descent():
    rng=np.random.default_rng(20801);p=rng.uniform(-20,20,(100,4,2));target=np.zeros((100,2))
    tv=np.tile([3.2,0.],(100,1));q=metric_correction(p,target,np.full(100,6.),C)
    command,alpha=allocate(tv[:,None],q,4.)
    derivative=np.sum(-q*(command-tv[:,None]),axis=(1,2))
    np.testing.assert_allclose(derivative,-np.sum(alpha*np.sum(q*q,axis=-1),axis=-1),atol=1e-10)
    assert (derivative<=1e-10).all() and (np.linalg.norm(command,axis=-1)<=4+1e-10).all()


def test_matches_inherited_angular_controller_on_reference_circle():
    angles=np.array([-.2,1.,2.7,4.8]);target=np.array([[7.,-2.]])
    p=target[:,None]+6*np.c_[np.cos(angles),np.sin(angles)][None]
    args=(p,target,np.array([[2.5,1.]]),np.array([6.]),C,np.zeros((1,1,2)))
    np.testing.assert_allclose(metric_budget_polar(*args),budget_polar(*args),atol=1e-12)


def test_translation_rotation_and_agent_permutation_equivariance():
    p=np.array([[[5.,3.],[-11.,6.],[-1.,-16.]]]);target=np.array([[1.,2.]]);tv=np.array([[2.,-.5]])
    a=.61;rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    shift=np.array([6.,-3.]);permutation=[2,0,1]
    old=metric_budget_polar(p,target,tv,np.array([6.]),C,np.zeros((1,1,2)))
    new=metric_budget_polar(p[:,permutation]@rotation.T+shift,target@rotation.T+shift,tv@rotation.T,np.array([6.]),C,np.zeros((1,1,2)))
    np.testing.assert_allclose(new,old[:,permutation]@rotation.T,atol=1e-12)


def test_degenerate_radius_still_gives_finite_bounded_command():
    command=metric_budget_polar(np.zeros((1,3,2)),np.zeros((1,2)),np.array([[3.2,0.]]),np.array([6.]),C,np.zeros((1,1,2)))
    assert np.isfinite(command).all() and (np.linalg.norm(command,axis=-1)<=4+1e-10).all()
