import unittest
import numpy as np
import copy
import torch
from .world import LocalBelief,LocalWorld,Conditions,fuse
from .policy import Policy,features
from .evaluate import score
from .predictor_r2 import EquivariantPredictor

class InformationTests(unittest.TestCase):
    def test_hidden_truth_discarded(self):
        a=LocalBelief(np.array([[.1,.2,.3]]));b=LocalBelief(a.sigma)
        mask=np.ones((1,3),bool);p=np.ones((1,3,2));v=np.ones_like(p)
        a.update(0,mask,p,v);b.update(0,mask,p,v)
        a.update(1,~mask,p*999,v*999);b.update(1,~mask,p*np.nan,v*np.nan)
        np.testing.assert_equal(a.feature,b.feature)
        for x,y in zip(a.query(1),b.query(1)):np.testing.assert_equal(x,y)

    def test_asynchronous_ages_and_initial_visibility(self):
        w=LocalWorld(np.arange(14930000,14930064),Conditions(loss=2))
        for _ in range(61):o=w.observe();w.step(o)
        self.assertTrue((np.ptp(o['local_age'],axis=1)>.5).all())
        self.assertTrue(np.isfinite(o['target']).all())
        self.assertTrue((o['local_var']>0).all())

    def test_invalid_message_not_fused(self):
        p=np.array([[[1.,1.],[999.,999.],[3.,3.]]]);v=p.copy();valid=np.array([[True,False,True]])
        mu,_,weight=fuse(p,v,np.ones((1,3)),np.zeros((1,3)),valid,'average')
        np.testing.assert_equal(mu,[[2.,2.]]);self.assertEqual(weight[0,1],0)

    def test_paired_sensor_noise_and_cache(self):
        a=LocalWorld([14930000],fusion='average');b=LocalWorld([14930000],fusion='precision')
        np.testing.assert_equal(a.noise,b.noise)
        o=a.observe();state=a.local.p.copy();self.assertIs(o,a.observe())
        np.testing.assert_equal(state,a.local.p)

    def test_actor_cannot_read_hidden_target(self):
        a=LocalWorld([14930000]);o=a.observe();a.time=12.;a.tick=120;a.cached=None
        b=copy.deepcopy(a);b.target+=30;b.tv-=20
        xa,na=features(a,a.observe());xb,nb=features(b,b.observe())
        np.testing.assert_equal(xa,xb);np.testing.assert_equal(na,nb)

    def test_same_parameter_budget_and_legal_actor_shape(self):
        torch.set_num_threads(1);torch.manual_seed(7);a=Policy(False)
        torch.manual_seed(7);b=Policy(True)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y)
        w=LocalWorld([14930000,14930001]);c,n=features(w,w.observe())
        d,v,att=a(torch.from_numpy(c),torch.from_numpy(n))
        self.assertEqual(d.logits.shape,(2,3,5));self.assertEqual(v.shape,(2,))
        torch.testing.assert_close(att.sum(-1),torch.ones((2,3)))

    def test_reacquisition_metric_and_no_preloss_credit(self):
        t=np.arange(1,301)*.1;mask=np.ones((300,1,3),bool);mask[120:180]=False
        geom=np.ones((300,1),bool);geom[120:]=False
        d=dict(time=t,geometry=geom,unsafe=np.zeros_like(geom),natural=np.ones_like(mask),mask=mask,
               age=np.zeros((300,1,3)),action=np.zeros((300,1,3),int))
        records,_=score(d,Conditions(loss=2),[0]);r=records[0]
        self.assertFalse(r['post_hold5']);self.assertFalse(r['recovery_within5'])
        self.assertAlmostEqual(r['reacquisition_delay'],4.)
        d['geometry'][200:]=True
        r=score(d,Conditions(loss=2),[0])[0][0]
        self.assertTrue(r['post_hold5']);self.assertTrue(r['recovery_within5'])
        self.assertAlmostEqual(r['recovery_delay'],2.1)

    def test_predictor_rotation_equivariance(self):
        torch.manual_seed(4);net=EquivariantPredictor(True)
        with torch.no_grad():net.mean.weight.normal_(0,.1)
        x=torch.randn(2,9,12);angle=.8;c=np.cos(angle);s=np.sin(angle);z=x.clone()
        for a,b in [(0,1),(2,3),(10,11)]:z[...,a]=c*x[...,a]-s*x[...,b];z[...,b]=s*x[...,a]+c*x[...,b]
        m,lv,_=net(x);mr,lvr,_=net(z);expected=m.clone()
        for a,b in [(0,1),(2,3)]:expected[...,a]=c*m[...,a]-s*m[...,b];expected[...,b]=s*m[...,a]+c*m[...,b]
        torch.testing.assert_close(mr,expected,atol=2e-6,rtol=1e-5);torch.testing.assert_close(lv,lvr)

if __name__=='__main__':unittest.main()
