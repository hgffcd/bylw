import copy,unittest
import numpy as np
from .guard import closing_limit,response_guard
from .world import ResponseWorld
from experiments.c150_native_search.task import Task


class GuardTests(unittest.TestCase):
    def test_measured_closing_speed_changes_braking(self):
        w=ResponseWorld([16300000],Task(scene='ring'));o=w.observe()
        o['p']=np.array([[[5.,0.],[20.,20.],[-20.,20.]]]);o['v']=np.zeros((1,3,2))
        o['target']=np.array([[0.,-20.]]);o['target_v']=np.zeros((1,2))
        o['obstacles']=np.array([[[0.,0.]]]);o['radii']=np.array([[3.]])
        command=np.zeros((1,3,2));command[0,0,0]=-1.
        slow=response_guard(command,o);o['v'][0,0,0]=-3.
        fast=response_guard(command,o)
        self.assertGreater(fast[0,0,0],slow[0,0,0]+1.)
        self.assertLessEqual(np.linalg.norm(fast,axis=-1).max(),4.+1e-9)

    def test_clearance_and_closing_monotonicity(self):
        self.assertGreater(closing_limit(.5,3),closing_limit(3,3))
        self.assertGreater(closing_limit(1,3),closing_limit(1,0))
        self.assertTrue(np.isfinite(closing_limit(-1,4)))

    def test_rotation_equivariance_away_from_boundary(self):
        w=ResponseWorld([16300001],Task(scene='corner'));o=w.observe();o['v'][:]=[3.,-1.]
        u=np.broadcast_to([2.,1.],(1,3,2)).copy();r=np.array([[0.,-1.],[1.,0.]])
        other=copy.deepcopy(o)
        for k in ['p','v','target','target_v','obstacles']:other[k]=other[k]@r.T
        np.testing.assert_allclose(response_guard(u,o)@r.T,response_guard(u@r.T,other),atol=1e-9)

    def test_stage_threshold_cache_exit_and_no_truth_leak(self):
        w=ResponseWorld([16300002],Task(scene='corner'));w.configure(slice(None),'staged_brake');w.observe()
        unseen=np.full((1,3,2),np.nan)
        w.sensor=lambda:(np.zeros((1,3),bool),unseen,unseen)
        for tick in range(1,15):
            w.time=round(tick*.1,8);o=w.observe()
            self.assertEqual(o['searcher'][0],-1);self.assertEqual(o['phase_mix'][0],0.)
        w.time=1.5;o=w.observe();self.assertGreaterEqual(o['searcher'][0],0)
        self.assertAlmostEqual(o['phase_mix'][0],.2)
        for _ in range(3):w.commands(w.observe())
        self.assertAlmostEqual(w.stage_mix[0],.2);self.assertEqual(w.stage_entries[0],1)
        b=copy.deepcopy(w);b.target+=20;b.tv-=10
        w.time=b.time=1.6
        for a,c in zip(w.commands(w.observe()),b.commands(b.observe())):np.testing.assert_equal(a,c)
        role=w.stage_role.copy()
        for tick in range(17,21):w.time=tick*.1;w.observe()
        self.assertAlmostEqual(w.stage_mix[0],1.)
        w.sensor=lambda:(np.array([[True,False,False]]),np.broadcast_to(w.target[:,None],(1,3,2)),np.zeros((1,3,2)))
        w.time=2.1;w.observe();np.testing.assert_equal(w.stage_role,role)
        for tick in range(22,27):w.time=round(tick*.1,8);w.observe()
        self.assertEqual(w.stage_role[0],-1);self.assertAlmostEqual(w.stage_mix[0],0.)

    def test_old_method_is_unchanged(self):
        from experiments.c150_native_search.task import SearchWorld
        for method,role in [('old_predict','predict_only'),('old_search','one_search')]:
            task=Task(scene='corner',role=role);a=SearchWorld([16300003],task);b=ResponseWorld([16300003],task)
            b.configure(slice(None),method)
            for tick in range(5):
                a.time=b.time=tick*.1
                for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):np.testing.assert_equal(x,y)

if __name__=='__main__':unittest.main()
