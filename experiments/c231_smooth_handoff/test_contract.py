import unittest
from unittest.mock import patch
import numpy as np
from experiments.c230_fixed3_time_match.planner import TimeProgressPlanner
from experiments.c231_smooth_handoff.planner import SmoothHandoffPlanner
from experiments.c230_fixed3_time_match.world import make_world as parent_world
from experiments.c231_smooth_handoff.world import make_world
from experiments.c231_smooth_handoff.design import ARMS,groups
from experiments.c231_smooth_handoff.scoring import gate,hold_score

class Contract(unittest.TestCase):
    def state(self):
        return dict(p=np.array([[[14.,0.],[0.,14.],[-14.,0.]]]),target=np.zeros((1,2)),
                    target_v=np.array([[2.,0.]]),active=np.ones((1,3),bool),
                    measured=np.ones(1,bool),phase_mix=np.zeros(1))

    def fixture(self):
        planner=SmoothHandoffPlanner(1,3)
        planner.bypass_on[0,0]=True;planner.bypass_applied[0,0]=True
        planner.bypass_start[0,0]=0.
        previous=np.zeros((1,3,2));previous[0,0]=[4.,0.]
        planner.handoff_output=(previous,np.ones(1,bool))
        base=np.zeros((1,3,2));base[0,0]=[0.,4.]
        def leave(o,b,t):
            planner.bypass_on[:]=False;planner.bypass_applied[:]=False
            return b.copy(),np.zeros(1,bool)
        return planner,base,leave

    def test_bounded_continuous_return_and_cache(self):
        p,b,leave=self.fixture();o=self.state()
        with patch.object(TimeProgressPlanner,'propose',side_effect=leave) as parent:
            first=p.propose(o,b,1.);np.testing.assert_array_equal(first[0][0,0],[4.,0.])
            self.assertIs(p.propose(o,b,1.),first);self.assertEqual(parent.call_count,1)
            mid=p.propose(o,b,1.3)[0];np.testing.assert_allclose(mid[0,0],[2.,2.],atol=1e-12)
            self.assertLessEqual(float(np.linalg.norm(mid,axis=-1).max()),4.)
            end=p.propose(o,b,1.7)[0];np.testing.assert_array_equal(end,b)
            self.assertEqual(p.handoff_entries[0,0],1)

    def test_loss_or_search_cancels_immediately(self):
        for key,value in [('measured',False),('phase_mix',.1)]:
            p,b,leave=self.fixture();o=self.state()
            with patch.object(TimeProgressPlanner,'propose',side_effect=leave):
                p.propose(o,b,1.);o[key][:]=value
                out,_=p.propose(o,b,1.1);np.testing.assert_array_equal(out,b)
                self.assertFalse(p.handoff_weight.any())

    def test_near_ring_and_lease_do_not_ramp(self):
        for near,time in [(True,1.),(False,6.1)]:
            p,b,leave=self.fixture();o=self.state()
            if near:o['p'][0,0]=[8.,0.]
            with patch.object(TimeProgressPlanner,'propose',side_effect=leave):
                out,_=p.propose(o,b,time)
                np.testing.assert_array_equal(out,b);self.assertFalse(p.handoff_entries.any())

    def test_no_previous_applied_path_no_ramp(self):
        p,b,leave=self.fixture();p.bypass_applied[:]=False
        with patch.object(TimeProgressPlanner,'propose',side_effect=leave):
            out,_=p.propose(self.state(),b,1.)
            np.testing.assert_array_equal(out,b);self.assertFalse(p.handoff_entries.any())

    def test_active_changes_or_new_bypass_cancel(self):
        p,b,leave=self.fixture();o=self.state()
        with patch.object(TimeProgressPlanner,'propose',side_effect=leave):
            p.propose(o,b,1.);o['active'][0,2]=False
            out,_=p.propose(o,b,1.1);np.testing.assert_array_equal(out,b)

    def test_frozen_variants_unchanged(self):
        kw=dict(n_pursuers=3,method='active_polar',n_obstacles=6,shape_profile='mixed',layout_profile='route_dense',recovery='budget_transit',search_mode='all',map_protocol='valid',opponent='routed')
        for variant in ['none','progress','matched']:
            a=parent_world('random_test',[22840000,22840001],control_variant=variant,**kw)
            b=make_world('random_test',[22840000,22840001],control_variant=variant,**kw)
            for key in ['p','target','z','obstacles','radii','routes','noise']:
                np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
            for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):
                np.testing.assert_array_equal(x,y)
        self.assertIsInstance(make_world('random_test',[22840000],control_variant='smooth',**kw).recovery_planner,SmoothHandoffPlanner)

    def test_fixed_comparison_gate_and_hold(self):
        self.assertEqual(sum(g['n'] for g in groups()),16)
        summary={str(w):dict(n=16,arms={a:dict(safe=16,moving_success=10 if a.startswith('smooth') else 8) for a in ARMS}) for w in [30,60]}
        self.assertTrue(gate(summary)['passed'])
        summary['30']['arms']['smooth_all']['moving_success']=9
        self.assertFalse(gate(summary)['passed'])
        one=np.ones(50,bool);one[20]=False
        self.assertFalse(hold_score(np.ones(50,bool),one,np.ones(50)*2,True)['moving_success'])

if __name__=='__main__':unittest.main()
