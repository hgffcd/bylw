import ast
from pathlib import Path
import unittest
import numpy as np
from experiments.c233_fixed3_final_unseen.design import groups,ARMS
from experiments.c233_fixed3_final_unseen.scoring import summarize,hold_score
from experiments.c222_unseen_maps.design import fingerprint

class Contract(unittest.TestCase):
    def test_matrix_and_balanced_order(self):
        gg=groups();self.assertEqual(sum(g['n'] for g in gg),32)
        self.assertEqual(sum(g['n']*len(ARMS) for g in gg)+16,144)
        self.assertEqual(len({g['seed']+i for g in gg for i in range(g['n'])}),32)
        for a in ARMS:self.assertEqual(sum(g['arm_order'][0]==a for g in gg),2)

    def test_native_runner_identical(self):
        r=Path(__file__).resolve().parents[2]
        self.assertEqual(ast.dump(ast.parse((r/'experiments/c225_search_removal/native_run.py').read_text())),
                         ast.dump(ast.parse((r/'experiments/c233_fixed3_final_unseen/native_run.py').read_text())))

    def test_paired_summary(self):
        rows=[]
        for a in ARMS:
            for seed in range(4):
                good=seed<2 if a=='ordinary_all' else seed<1
                rows.append(dict(arm=a,seed=seed,window_s=30,moving_success=good,success=good,safe=True,
                    had_loss=True,search_deployed=a!='ordinary_predict',reacquired=True,capture_after_last_reacquisition=good,total_loss_s=2.))
        s=summarize(rows)['30'];self.assertEqual(s['n'],4)
        c=s['contrasts']['ordinary_all_vs_ordinary_predict'];self.assertEqual(c['delta_pp'],25.)
        self.assertEqual(c['gains'],[1]);self.assertEqual(c['paired_exact_two_sided_p'],1.)

    def test_fingerprint_and_visible_hold(self):
        p=np.array([[0.,0.],[2.,3.],[4.,-1.]]);r=np.array([.4,.5,.6])
        self.assertEqual(fingerprint(p,r),fingerprint((p+2)[::-1],r[::-1]))
        vis=np.ones(50,bool);vis[20]=False
        self.assertFalse(hold_score(np.ones(50,bool),vis,np.ones(50)*2,True)['moving_success'])

if __name__=='__main__':unittest.main()
