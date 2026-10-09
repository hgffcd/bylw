"""C200: change observed polar speed saturation only, keep search and safety."""
import numpy as np
from experiments.c175_multi_fault_reconfig.world import unit,polar,limit,norm
from experiments.c191_route_validity.world import ValidRouteWorld
from experiments.c198_planner_input_audit.world import make_world as base_world
from experiments.c200_feedforward_budget.controller import budget_polar


class BudgetRouteWorld(ValidRouteWorld):
    def _core_commands(self,o,actions=None):
        c=self.cfg;n=len(self.seeds);delta=o['p']-o['target'][:,None];er=unit(delta);et=np.stack([-er[...,1],er[...,0]],-1)
        command=polar(o['p'],o['v'],o['target'],o['target_v'],self.control_radii,c)
        acc=o['omega'][:,None]*np.stack([-o['target_v'][:,1],o['target_v'][:,0]],-1)
        command+=self.compensations[:,None,None]*acc[:,None]
        allocated=budget_polar(o['p'],o['target'],o['target_v'],self.control_radii,c,self.compensations[:,None,None]*acc[:,None])
        command=np.where(o['measured'][:,None,None],allocated,command)
        yaw_goal=np.arctan2(-delta[...,1],-delta[...,0]);altitude=np.full((n,self.n_pursuers),3.)
        active=(o['searcher']>=0)&(self.role_modes>0)
        if active.any():
            # The two supporters keep their own stored bearings around a belief
            # whose feedforward decays with age. They cannot guard hidden truth.
            feed=o['target_v']*np.exp(-o['age'][:,None]/3.)
            goals=o['target'][:,None]+self.control_radii[:,None,None]*np.stack([np.cos(o['guard_angle']),np.sin(o['guard_angle'])],-1)
            guard=feed[:,None]+.8*(goals-o['p'])
            # Search sweeps a flank of the predicted region, choosing an obstacle
            # clear side using the public map and the current belief only.
            angle=o['guard_angle'];phase=np.clip(o['age'],0,5)*.55
            left=angle+phase[:,None];right=angle-phase[:,None]
            rr=self.control_radii[:,None]+np.minimum(o['age'],4)[:,None]*.35
            left_goal=o['target'][:,None]+rr[...,None]*np.stack([np.cos(left),np.sin(left)],-1)
            right_goal=o['target'][:,None]+rr[...,None]*np.stack([np.cos(right),np.sin(right)],-1)
            lc=(norm(left_goal[:,:,None]-o['obstacles'][:,None])-o['radii'][:,None]).min(-1)
            rc=(norm(right_goal[:,:,None]-o['obstacles'][:,None])-o['radii'][:,None]).min(-1)
            choose=lc>=rc
            goal=np.where(choose[...,None],left_goal,right_goal)
            search=feed[:,None]+.9*(goal-o['p'])
            selector=np.arange(self.n_pursuers)[None]==o['searcher'][:,None]
            selector=np.where(self.role_modes[:,None]==2,True,selector)
            command=np.where(active[:,None,None],np.where(selector[...,None],search,guard),command)
            scan=.6*np.sin(1.6*o['age'][:,None]+np.arange(self.n_pursuers)[None]*.6)
            yaw_goal+=np.where(active[:,None]&selector,scan,0.)
            altitude=np.where(active[:,None]&selector,3.5,np.where(active[:,None],3+np.linspace(-.15,.15,self.n_pursuers)[None],3.))
        if actions is not None:
            radial=np.array([0,1,-1,0,0,0,0,0])[actions]
            tangent=np.array([0,0,0,1,-1,0,0,0])[actions]
            vertical=np.array([0,0,0,0,0,1,-1,0])[actions]
            command+=.6*(radial[...,None]*er+tangent[...,None]*et)
            command=np.where((actions==7)[...,None],.5*command,command)
            altitude+=.3*vertical
        altitude=np.clip(altitude,2.4,3.6)
        if self.slot_control:
            # Legal CBC-TP-style short-horizon extrapolation. It uses only
            # the shared position/velocity estimate and loss age.
            slot_center = o['target']
            if self.method == 'predictive_hysteretic':
                horizon = np.clip(0.5 * o['age'], 0.0, 1.0)
                slot_center = slot_center + horizon[:, None] * o['target_v']
            slot_goal=slot_center[:,None]+self.control_radii[:,None,None]*np.stack(
                (np.cos(self.slot_angles),np.sin(self.slot_angles)),axis=-1
            )
            slot_delta=np.nan_to_num(slot_goal-o['p'])
            # Use a bounded proportional reference toward the reassigned slot.
            # The target-velocity feedforward keeps the transition from
            # fighting the evader motion; the safety and speed projections
            # below still own the final command.
            correction=self.slot_gain*slot_delta
            mix=np.clip((self.time-self.reconfig_start)/self.reconfig_duration,0.,1.)
            feedforward = 0. if self.method == 'no_feedforward' else 1.
            slot_command=limit(feedforward*o['target_v'][:,None]+correction,c.pursuer_speed)
            slot_command=limit((1.-mix[:,None,None])*command+mix[:,None,None]*slot_command,c.pursuer_speed)
            # A searcher retains its search trajectory; the remaining members
            # absorb the new angular slots and restore the closed formation.
            apply=self.active & (o['searcher']<0)[:,None]
            if self.method in ('gated_dynamic', 'hysteretic_dynamic', 'predictive_hysteretic'):
                apply &= self.gate_active[:, None]
            command=np.where(apply[:,:,None],slot_command,command)
        command=self.safety_filter(limit(command,c.pursuer_speed),o)
        command=np.where(self.active[...,None],command,0.0)
        return command,yaw_goal,altitude


def make_world(*args,recovery='routed_transit',**kwargs):
    w=base_world(*args,recovery='routed_transit' if recovery=='budget_transit' else recovery,**kwargs)
    if recovery=='budget_transit':
        # Same instance state and __init__; copy.copy used by active compaction
        # retains this subclass and dispatches _core_commands on that view.
        assert isinstance(w,ValidRouteWorld) and w.n_pursuers==3
        w.__class__=BudgetRouteWorld
        w.requested_recovery='budget_transit'
    return w
