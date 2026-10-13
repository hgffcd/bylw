import ast
from pathlib import Path
import unittest
import numpy as np
from experiments.c232_reconfiguration_control.design import groups
from experiments.c232_reconfiguration_control.scoring import hold_score
from experiments.c210_radial_damping.world import make_world

class Contract(unittest.TestCase):
    def test_full_registered_cohort(self):
        gg=groups();self.assertEqual(len(gg),12)
        for s in ['4to3','5to4','5to3']:
            self.assertEqual(sum(g['n'] for g in gg if g['stratum']==s),16)
        self.assertEqual(len({g['seed']+i for g in gg for i in range(g['n'])}),48)

    def test_native_runner_identical(self):
        root=Path(__file__).resolve().parents[2]
        a=ast.dump(ast.parse((root/'experiments/c222_unseen_maps/native_run.py').read_text()))
        b=ast.dump(ast.parse((root/'experiments/c232_reconfiguration_control/native_run.py').read_text()))
        self.assertEqual(a,b)

    def test_prefault_same_and_retired_body_kept(self):
        for n in [4,5]:
            kw=dict(n_pursuers=n,n_obstacles=6,shape_profile='mixed',layout_profile='route_dense',recovery='radial_damped_sector',map_protocol='valid',opponent='routed')
            a=make_world('random_test',[22250000],method='active_polar',**kw)
            b=make_world('random_test',[22250000],method='retained_slots',**kw)
            self.assertFalse(a.slot_control);self.assertFalse(b.slot_control)
            for key in ['p','target','z','obstacles','radii','routes','noise']:
                np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
            for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):
                np.testing.assert_array_equal(x,y)
            a.withdraw(0);b.withdraw(0)
            self.assertFalse(a.slot_control);self.assertTrue(b.slot_control)
            self.assertFalse(a.active[0,0]);self.assertFalse(b.active[0,0])
            self.assertEqual(a.p.shape[1],n);self.assertEqual(b.p.shape[1],n)

    def test_visibility_is_explicit_secondary(self):
        visible=np.ones(50,bool);visible[25]=False
        result=hold_score(np.ones(50,bool),visible,np.full(50,2.),True)
        self.assertFalse(result['moving_success'])
        self.assertTrue(hold_score(np.ones(50,bool),np.ones(50,bool),np.full(50,2.),True)['moving_success'])
        self.assertFalse(hold_score(np.ones(50,bool),np.ones(50,bool),np.full(50,.5),True)['moving_success'])

if __name__=='__main__':unittest.main()
