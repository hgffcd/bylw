import unittest,copy
import numpy as np
from .task import SearchWorld,Task,geometry3,visibility3
from .policy import features

class TaskTests(unittest.TestCase):
    def test_geometry_tolerance_not_just_proximity(self):
        angles=np.deg2rad([0,110,240]);p=np.array([5.5,6,6.5])[:,None]*np.stack([np.cos(angles),np.sin(angles)],-1)
        valid,*_=geometry3(p[None],np.array([[2.8,3,3.3]]),np.zeros((1,2)),np.array([3.]))
        self.assertTrue(valid[0])
        bad=np.array([[[5,0],[6,.5],[7,-.5]]],float)
        self.assertFalse(geometry3(bad,np.ones((1,3))*3,np.zeros((1,2)),np.array([3.]))[0][0])

    def test_any_real_sighting_refreshes_shared_belief(self):
        w=SearchWorld([16000000]);mask=np.array([[False,True,False]])
        p=np.array([[[np.nan,np.nan],[4.,5.],[np.nan,np.nan]]]);v=np.array([[[np.nan,np.nan],[1.,0.],[np.nan,np.nan]]])
        w.sensor=lambda:(mask,p,v);o=w.observe()
        np.testing.assert_equal(o['target'],[[4,5]]);self.assertEqual(o['age'][0],0)
        self.assertTrue(o['measured'][0])

    def test_hidden_truth_cannot_change_search_command(self):
        w=SearchWorld([16000001]);w.observe();w.time=1.;w.cached=None
        mask=np.zeros((1,3),bool);unseen=np.full((1,3,2),np.nan);w.sensor=lambda:(mask,unseen,unseen)
        w.observe();w.time=1.1;w.cached=None;b=copy.deepcopy(w);b.target+=30;b.tv-=15
        x=w.observe();y=b.observe()
        for u,v in zip(w.commands(x),b.commands(y)):np.testing.assert_equal(u,v)
        np.testing.assert_equal(features(w,x),features(b,y));self.assertGreaterEqual(w.searcher[0],0)

    def test_height_actions_and_brake_reach_commands(self):
        w=SearchWorld([16000002]);o=w.observe();u,y,z=w.commands(o,np.zeros((1,3),int))
        _,_,zup=w.commands(o,np.full((1,3),5));self.assertTrue((zup>z).all())
        slow,_,_=w.commands(o,np.full((1,3),7));self.assertLess(np.linalg.norm(slow),np.linalg.norm(u))

    def test_target_leaves_final_route_waypoint(self):
        w=SearchWorld([16000002]);w.route_index[:]=2;w.target=w.routes[:,2].copy()
        u=w.evader_command();self.assertTrue(w.route_done.all());self.assertGreater(np.linalg.norm(u),2.)

    def test_loss_event_clock_and_post_reacquisition_hold(self):
        from .metrics import score
        t=80;theta=np.arange(3)*2*np.pi/3
        p=6*np.stack([np.cos(theta),np.sin(theta)],-1)
        data=dict(time=np.arange(1,t+1)*.1,p=np.broadcast_to(p,(t,1,3,2)),z=np.full((t,1,3),3.),
            target=np.zeros((t,1,2)),target_z=np.full((t,1),3.),geometry=np.ones((t,1),bool),
            mask=np.ones((t,1,3),bool),unsafe=np.zeros((t,1),bool),failures=np.zeros((t,1,7),bool))
        data['mask'][10:20]=False
        episodes,summary=score(data,[0]);event=episodes[0]['loss_events'][0]
        self.assertEqual(event['start_s'],1.1);self.assertEqual(event['duration_s'],1.)
        self.assertEqual(event['end_tick'],20);self.assertEqual(summary['post_loss_capture5'],1.)
        self.assertEqual(summary['legacy_geometry_safe_capture5'],0.)

    def test_fov_range_and_occlusion_are_distinct(self):
        w=SearchWorld([16000002]);p=np.array([[[-10.,0.],[-21.,0.],[-10.,3.]]]);z=np.full((1,3),3.)
        target=np.zeros((1,2));obstacles=np.array([[[-5.,0.]]]);radii=np.array([[1.]])
        yaw=np.array([[0.,0.,np.arctan2(-3.,10.)]])
        seen=visibility3(p,z,target,np.array([3.]),yaw,obstacles,radii,w.cfg)
        np.testing.assert_equal(seen,[[False,False,True]])
        yaw[0,2]+=np.pi
        self.assertFalse(visibility3(p,z,target,np.array([3.]),yaw,obstacles,radii,w.cfg).any())

    def test_search_role_locks_and_shared_reacquisition_exits(self):
        w=SearchWorld([16000002]);w.observe();missing=np.full((1,3,2),np.nan)
        w.sensor=lambda:(np.zeros((1,3),bool),missing,missing)
        for t in [.1,.2]:w.time=t;w.observe()
        role=w.searcher.copy();self.assertGreaterEqual(role[0],0)
        w.p=w.p[:,::-1].copy();w.time=.3;w.observe();np.testing.assert_equal(w.searcher,role)
        w.sensor=lambda:(np.array([[True,False,False]]),np.broadcast_to(w.target[:,None],(1,3,2)),np.zeros((1,3,2)))
        w.time=.4;w.observe();np.testing.assert_equal(w.searcher,role)
        w.time=.5;w.observe();self.assertEqual(w.searcher[0],-1);self.assertEqual(w.age[0],0)

if __name__=='__main__':unittest.main()
