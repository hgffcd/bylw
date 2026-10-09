"""C204: match a routed destination to its estimated travel time.

The bypass state machine below is frozen from C195; only destination planning
changes. CV is a public-estimate extrapolation, not knowledge of the route.
"""
import numpy as np
from experiments.c184_route_recovery.planner import VisibilityRouter
from experiments.c193_nearfield_guard.planner import visible_edges, TransitRecoveryPlanner
from experiments.c195_sector_return.planner import sector_candidates

# Existing replanning period and bypass lease define the forecast grid.
HORIZONS = np.arange(1, 11, dtype=float) * .6


def time_matched_plan(p, member, center, velocity, obstacles, radii):
    offsets,angle=sector_candidates(p,member,center)
    if not len(offsets):return None
    horizons=np.repeat(HORIZONS,len(offsets))
    offsets=np.tile(offsets,len(HORIZONS))
    predicted=np.clip(center+horizons[:,None]*velocity,-46.,46.)
    goals=predicted+6.*np.c_[np.cos(angle+offsets),np.sin(angle+offsets)]
    envelopes=np.asarray(radii)+1.25
    legal=(np.abs(goals)<48.5).all(1)
    if len(obstacles):
        legal &= (np.linalg.norm(goals[:,None]-obstacles,axis=-1)-envelopes).min(1)>=0
    if not legal.any():return None
    goals,horizons,offsets=goals[legal],horizons[legal],offsets[legal]
    router=VisibilityRouter(np.r_[obstacles,center[None]],np.r_[envelopes,3.2],goals)
    length,paths=router.paths_from(p[member])
    # Optimistic travel time: acceleration, safety projection and target turns
    # can delay arrival. Neither this check nor its fallback certifies capture.
    eta=length/4.
    reachable=np.isfinite(eta)&(eta<=horizons+1e-9)
    finite=np.flatnonzero(np.isfinite(eta))
    if not len(finite):return None
    if reachable.any():
        choices=np.flatnonzero(reachable)
        # First horizon satisfying the deadline; retain the old angular cost
        # as a secondary tie breaker, converted from metres to seconds.
        best=min(choices,key=lambda j:(horizons[j],eta[j]+.125*abs(offsets[j]),int(j)))
    else:
        # Within the bounded lease there may be no predicted interception.
        # Follow the least late routed goal, and replan at the original .6 s.
        best=min(finite,key=lambda j:(eta[j]-horizons[j],eta[j]+.125*abs(offsets[j]),int(j)))
    return dict(path=paths[best],horizon=float(horizons[best]),eta=float(eta[best]),
                offset=float(offsets[best]),deadline_met=bool(reachable[best]),
                route_length=float(length[best]),goal=goals[best].copy())


def time_matched_path(*args,**kwargs):
    plan=time_matched_plan(*args,**kwargs)
    return None if plan is None else plan['path']


class TimeMatchedBypassPlanner(TransitRecoveryPlanner):
    def __init__(self,batch,agents):
        super().__init__(batch,agents)
        self.bypass_on=np.zeros((batch,agents),bool)
        self.bypass_entries=np.zeros((batch,agents),int)
        self.bypass_calls=np.zeros((batch,agents),int)
        self.bypass_no_plan=np.zeros((batch,agents),int)
        self.bypass_stall=np.zeros((batch,agents))
        self.bypass_start=np.zeros((batch,agents))
        self.bypass_next=np.zeros((batch,agents))
        self.bypass_cooldown=np.zeros((batch,agents))
        self.bypass_clear=np.zeros((batch,agents))
        self.bypass_paths=[[None]*agents for _ in range(batch)]
        self.bypass_clock=None
        self.bypass_output=None
        self.bypass_applied=np.zeros((batch,agents),bool)

    def propose(self,o,base,time):
        if self.bypass_clock==time and self.bypass_output is not None:return self.bypass_output
        dt=0. if self.bypass_clock is None else max(0.,time-self.bypass_clock)
        self.bypass_clock=time
        out,changed=super().propose(o,base,time)
        out,changed=out.copy(),changed.copy()
        self.bypass_applied[:]=False
        for i in range(len(base)):
            active=np.flatnonzero(o['active'][i])
            eligible=len(active) in (3,4) and len(active)<self.agents and bool(o['measured'][i]) and np.linalg.norm(o['target_v'][i])>1.
            if not eligible:
                self.bypass_on[i]=False;self.bypass_stall[i]=0.;self.bypass_clear[i]=0.
                continue
            retired=o['p'][i,~o['active'][i]]
            obstacles=np.r_[o['obstacles'][i],retired]
            radii=np.r_[o['radii'][i],np.full(len(retired),.6)]
            center,velocity=o['target'][i],o['target_v'][i]
            for j in active:
                p=o['p'][i,j];delta=p-center;radius=np.linalg.norm(delta)
                clearance=(np.linalg.norm(p-obstacles,axis=-1)-radii).min()
                base_speed=np.linalg.norm(base[i,j]);actual_speed=np.linalg.norm(o['v'][i,j])
                stalled=radius>12. and clearance<3. and base_speed<1. and actual_speed<1.
                self.bypass_stall[i,j]=self.bypass_stall[i,j]+dt if stalled else 0.
                if not self.bypass_on[i,j] and time>=self.bypass_cooldown[i,j] and self.bypass_stall[i,j]>=.4:
                    self.bypass_on[i,j]=True;self.bypass_entries[i,j]+=1
                    self.bypass_start[i,j]=time;self.bypass_next[i,j]=time;self.bypass_clear[i,j]=0.
                if not self.bypass_on[i,j]:continue
                direct=visible_edges(p[None],(p+1.2*base[i,j])[None],obstacles,radii+1.25)[0,0]
                closing=-np.dot(base[i,j]-velocity,delta/max(radius,1e-9))
                clear=bool(direct and base_speed>1.2 and closing>.2)
                self.bypass_clear[i,j]=self.bypass_clear[i,j]+dt if clear else 0.
                if radius<8.5 or self.bypass_clear[i,j]>=.6 or time-self.bypass_start[i,j]>=6.:
                    self.bypass_on[i,j]=False;self.bypass_cooldown[i,j]=time+1.5
                    self.bypass_stall[i,j]=0.;self.bypass_paths[i][j]=None
                    continue
                if time>=self.bypass_next[i,j]-1e-8:
                    self.bypass_calls[i,j]+=1
                    self.bypass_paths[i][j]=time_matched_path(o['p'][i,active],int(np.flatnonzero(active==j)[0]),center,velocity,obstacles,radii)
                    self.bypass_next[i,j]=time+.6
                    if self.bypass_paths[i][j] is None:self.bypass_no_plan[i,j]+=1
                path=self.bypass_paths[i][j]
                if path is None:continue
                while len(path)>1 and np.linalg.norm(path[0]-p)<.7:
                    path=path[1:]
                self.bypass_paths[i][j]=path
                command=4.*(path[0]-p)/max(np.linalg.norm(path[0]-p),1e-9)
                if len(path)==1:
                    command=velocity+.9*(path[0]-p)
                    command*=min(1.,4./max(np.linalg.norm(command),1e-9))
                out[i,j]=command;changed[i]=True;self.bypass_applied[i,j]=True
        self.bypass_output=out,changed
        return self.bypass_output
