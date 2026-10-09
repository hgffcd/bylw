import numpy as np
from experiments.c195_sector_return.planner import sector_candidates, sector_path, SectorBypassPlanner
from experiments.c192_bounded_bypass.test_bypass import observation
from experiments.c184_route_recovery.planner import visible_edges


def test_sector_preserves_order_and_never_worsens_adjacent_gap_energy():
    rng=np.random.default_rng(195)
    for n in (3,4):
        for _ in range(200):
            angles=np.sort(rng.uniform(-np.pi,np.pi,n))
            p=6*np.c_[np.cos(angles),np.sin(angles)]
            for j in range(n):
                offsets,angle=sector_candidates(p,j,np.zeros(2))
                relative=(np.delete(angles,j)-angle)%(2*np.pi)
                left=2*np.pi-relative.max();right=relative.min()
                assert len(offsets)>0
                assert np.all(left+offsets>0) and np.all(right-offsets>0)
                assert np.all((left+offsets)**2+(right-offsets)**2<=left**2+right**2+1e-9)


def test_regular_formation_rejects_angularly_harmful_endpoints():
    for n in (3,4):
        angles=np.arange(n)*2*np.pi/n
        p=6*np.c_[np.cos(angles),np.sin(angles)]
        offsets,_=sector_candidates(p,0,np.zeros(2))
        np.testing.assert_allclose(offsets,0,atol=1e-9)


def test_route_is_collision_free_and_moves_toward_larger_gap():
    center=np.array([10.,0.]);start=np.array([-8.,0.])
    angles=np.deg2rad([180.,240.,60.])
    p=center+6*np.c_[np.cos(angles),np.sin(angles)];p[0]=start
    obs=np.array([[0.,0.]]);radii=np.array([2.]);velocity=np.array([2.,0.])
    path=sector_path(p,0,center,velocity,obs,radii)
    assert path is not None
    chain=np.r_[start[None],path]
    for a,b in zip(chain[:-1],chain[1:]):
        assert visible_edges(a[None],b[None],np.r_[obs,center[None]],np.r_[radii+1.25,3.2])[0,0]
    offset=(np.arctan2(*(path[-1]-center-1.2*velocity)[::-1])-np.pi+np.pi)%(2*np.pi)-np.pi
    assert offset<=1e-8


def test_only_selected_proposal_changes_loss_hands_back_and_lease_expires():
    o=observation();o['p'][0,2,1]=2.
    base=np.ones((1,5,2))*2; base[0,0]=0
    planner=SectorBypassPlanner(1,5)
    for tick in range(6):out,changed=planner.propose(o,base,tick*.1)
    assert changed[0] and planner.bypass_applied[0,0]
    np.testing.assert_array_equal(out[0,1:],base[0,1:])
    assert np.linalg.norm(out[0,0])<=4+1e-8
    for tick in range(6,70):out,changed=planner.propose(o,base,tick*.1)
    assert not planner.bypass_on.any() and not changed.any()
    o['measured'][:]=False
    out,changed=planner.propose(o,base,7.)
    np.testing.assert_array_equal(out,base)


def test_permutation_and_rotation_do_not_change_selected_offsets():
    p=np.array([[14.,1.],[0.,7.],[-5.,-4.],[7.,-8.]])
    center=np.zeros(2);offsets,_=sector_candidates(p,0,center)
    perm=[3,1,0,2];other,_=sector_candidates(p[perm],2,center)
    np.testing.assert_allclose(offsets,other,atol=1e-12)
    t=.71;rotation=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    other,_=sector_candidates(p@rotation.T,0,center)
    np.testing.assert_allclose(offsets,other,atol=1e-12)
