"""Obstacle-feasible adversary variant. Slowing is explicit, never teleporting.

The nominal largest-gap intention is retained. Near obstacles or boundaries,
choose a bounded-heading velocity by a 2 s acceleration-limited look-ahead.
This is a task validity repair, not a proposed pursuer algorithm.
"""
import numpy as np
from .planar import World,norm,limit,wrap


class PreviewWorld(World):
    def evader_command(self):
        old_heading=self.heading.copy()
        nominal=super().evader_command();c=self.cfg
        if c.evader!='smooth_gap':raise ValueError('preview opponent requires smooth_gap')
        trigger=np.any(np.abs(self.target)>c.arena_half-7.,axis=1)
        if c.obstacles:
            trigger|=(norm(self.target[:,None]-self.obstacles)-self.radii).min(1)<7.
        if not trigger.any():return nominal
        ids=np.flatnonzero(trigger);dt=.1;horizon=20
        slew=c.target_turn_rate*c.dt
        offsets=np.array([-1.,-.5,0.,.5,1.])*slew
        headings=np.concatenate([self.heading[ids,None],old_heading[ids,None]+offsets],axis=1)
        headings=np.repeat(headings,5,axis=1)
        scales=np.tile([1.,.75,.5,.25,0.],6)
        commands=c.pursuer_speed*c.evader_ratio*scales[None,:,None]*np.stack([np.cos(headings),np.sin(headings)],axis=-1)
        pos=np.repeat(self.target[ids,None],commands.shape[1],axis=1)
        vel=np.repeat(self.tv[ids,None],commands.shape[1],axis=1)
        clearance=np.full(pos.shape[:2],np.inf)
        for _ in range(horizon):
            vel+=dt*limit(c.velocity_kp*(commands-vel),c.acceleration)
            pos+=dt*vel
            current=(c.arena_half-.7-np.abs(pos)).min(-1)
            if c.obstacles:
                obstacle_clearance=(norm(pos[:,:,None]-self.obstacles[ids,None])-self.radii[ids,None]-.7).min(-1)
                current=np.minimum(current,obstacle_clearance)
            clearance=np.minimum(clearance,current)
        feasible=clearance>=0
        deviation=np.sum((commands-nominal[ids,None])**2,axis=-1)
        # Feasibility precedes nominal-command fidelity; if none are feasible,
        # report the resulting physical failure rather than resetting a target.
        score=np.where(feasible,deviation,1e5+1e4*np.maximum(-clearance,0)+deviation)
        choice=np.argmin(score,axis=1)
        nominal[ids]=commands[np.arange(len(ids)),choice]
        self.heading[ids]=wrap(headings[np.arange(len(ids)),choice])
        return nominal
