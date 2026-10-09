import unittest
import numpy as np
from experiments.c150_native_search.task import Task
from .world import HiddenBranchWorld


class HiddenBranchTests(unittest.TestCase):
    def test_initial_observation_and_hidden_branch(self):
        a=HiddenBranchWorld([16500000],Task(scene='corner'))
        b=HiddenBranchWorld([16500001],Task(scene='corner'))
        self.assertTrue(a.observe()['measured'][0]);self.assertTrue(b.observe()['measured'][0])
        self.assertIn(a.branch[0],[0,1]);self.assertIn(b.branch[0],[0,1])

    def test_first_waypoint_is_common_to_both_routes(self):
        w=HiddenBranchWorld([16500000,16500001,16600001],Task(scene='corner'),1.6)
        common=np.asarray([w.routes[i,0]-w.target[i] for i in range(len(w.seeds))])
        self.assertTrue(np.allclose(w.branch_routes[:,0]-w.target,common,atol=1e-7))

    def test_branch_does_not_enter_observation(self):
        a=HiddenBranchWorld([16500000],Task(scene='corner'));b=HiddenBranchWorld([16500000],Task(scene='corner'))
        self.assertEqual(a.branch[0],b.branch[0])
        a.branch[:]=1-b.branch
        oa=a.observe();ob=b.observe()
        for key in ['target','target_v','age','measured','mask','searcher','guard_angle']:
            np.testing.assert_equal(oa[key],ob[key])

    def test_branch_does_not_enter_policy_features_or_commands(self):
        from experiments.c150_native_search.policy import features
        a=HiddenBranchWorld([16500003],Task(scene='corner'));b=HiddenBranchWorld([16500003],Task(scene='corner'))
        a.branch[:]=1-b.branch
        oa=a.observe();ob=b.observe()
        np.testing.assert_equal(features(a,oa),features(b,ob))
        for x,y in zip(a.commands(oa),b.commands(ob)):np.testing.assert_equal(x,y)

    def test_branch_route_has_distinct_continuation(self):
        self.assertFalse(np.allclose(HiddenBranchWorld.BRANCH_LOCAL[0,1],HiddenBranchWorld.BRANCH_LOCAL[1,1]))

if __name__=='__main__':unittest.main()
