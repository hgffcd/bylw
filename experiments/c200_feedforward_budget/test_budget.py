import numpy as np
from experiments.c200_feedforward_budget.controller import allocate
from experiments.c200_feedforward_budget.world import make_world
from experiments.c198_planner_input_audit.world import make_world as old_world
from experiments.c145_feasibility_reset.planar import limit


def test_feasible_feed_and_uniform_correction_with_speed_limit():
    rng=np.random.default_rng(200)
    feed=rng.normal(size=(1000,2))*4;q=rng.normal(size=(1000,2))*10
    command,a=allocate(feed,q)
    assert (np.linalg.norm(command,axis=-1)<=4+1e-10).all()
    np.testing.assert_allclose(command-limit(feed,4),a[:,None]*q,atol=1e-12)
    assert ((a>=0)&(a<=1)).all()
    partial=a<1-1e-8
    assert (np.linalg.norm(limit(feed,4)[partial]+(a[partial]+1e-5)[:,None]*q[partial],axis=-1)>4).all()


def test_translation_retained_during_large_lateral_correction():
    f=np.array([3.2,0.]);q=np.array([0.,14.])
    command,_=allocate(f,q)
    np.testing.assert_allclose(command,[3.2,2.4],atol=1e-12)
    assert limit(f+q,4)[0]<1.


def test_unsaturated_and_zero_feedback_cases():
    np.testing.assert_allclose(allocate(np.array([1.,0.]),np.array([0.,1.]))[0],[1.,1.])
    np.testing.assert_allclose(allocate(np.array([3.2,0.]),np.zeros(2))[0],[3.2,0.])
    np.testing.assert_allclose(allocate(np.zeros(2),np.array([6.,8.]))[0],[2.4,3.2])


def test_map_opponent_and_loss_search_commands_remain_identical():
    args=dict(n_pursuers=3,method='active_polar',n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')
    a=old_world('random_test',[19640019],recovery='routed_transit',**args)
    b=make_world('random_test',[19640019],recovery='budget_transit',**args)
    for name in ['p','target','obstacles','radii','routes','active']:
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    np.testing.assert_array_equal(a.evader_command(),b.evader_command())
    o=a.observe();o=dict(o,measured=np.zeros(1,bool),searcher=np.zeros(1,int),age=np.ones(1))
    aa=a.commands(o,np.zeros((1,3),int));bb=b.commands(o,np.zeros((1,3),int))
    for x,y in zip(aa,bb):np.testing.assert_array_equal(x,y)


def test_rigid_rotation_preserves_allocation():
    t=.8;r=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    f=np.array([2.,1.]);q=np.array([-2.,8.]);a,alpha=allocate(f,q);b,beta=allocate(f@r.T,q@r.T)
    np.testing.assert_allclose(a@r.T,b,atol=1e-12);np.testing.assert_allclose(alpha,beta,atol=1e-12)
