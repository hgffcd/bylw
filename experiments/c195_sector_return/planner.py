"""C195: constrain bypass endpoints by the observed adjacent angular gaps.

Endpoint consistency is a local planning heuristic, not a dynamic invariant.
No hidden target route or true unobserved target state is used.
"""
import numpy as np
from experiments.c184_route_recovery.planner import VisibilityRouter
from experiments.c193_nearfield_guard.planner import visible_edges, TransitRecoveryPlanner


def sector_candidates(p, member, center):
    """Return offsets that preserve cyclic order and do not raise gap energy.

    Holding other bearings fixed, moving this bearing by d changes the two
    adjacent gaps from (left,right) to (left+d,right-d). Their squared-sum
    minimum is at (right-left)/2. The check is at the observed center only.
    """
    delta=np.asarray(p)-center
    theta=np.arctan2(delta[:,1],delta[:,0])
    angle=float(theta[member])
    relative=(np.delete(theta,member)-angle)%(2*np.pi)
    if np.any((relative<1e-8)|(relative>2*np.pi-1e-8)):
        return np.empty(0),angle
    left=2*np.pi-relative.max();right=relative.min()
    midpoint=.5*(right-left)
    # Existing 0,+/-0.25 rad choices plus one gap-balancing destination.
    # Cap its change by half a nominal team gap, with no seed/identity rule.
    extra=np.clip(midpoint,-np.pi/len(p),np.pi/len(p))
    offsets=np.unique(np.r_[0.,-.25,.25,extra])
    potential=(left+offsets)**2+(right-offsets)**2
    legal=(offsets>-left+1e-8)&(offsets<right-1e-8)
    legal &= potential<=left**2+right**2+1e-10
    return offsets[legal],angle


def sector_path(p, member, center, velocity, obstacles, radii):
    offsets,angle=sector_candidates(p,member,center)
    if not len(offsets):return None
    predicted=np.clip(center+1.2*velocity,-46.,46.)
    goals=predicted+6.*np.c_[np.cos(angle+offsets),np.sin(angle+offsets)]
    envelopes=radii+1.25
    legal=(np.abs(goals)<48.5).all(1)
    legal &= (np.linalg.norm(goals[:,None]-obstacles,axis=-1)-envelopes).min(1)>=0
    if not legal.any():return None
    router=VisibilityRouter(np.r_[obstacles,center[None]],np.r_[envelopes,3.2],goals[legal])
    costs,paths=router.paths_from(p[member])
    costs+=.5*np.abs(offsets[legal])
    if not np.isfinite(costs).any():return None
    return paths[int(np.argmin(costs))]


class SectorBypassPlanner(TransitRecoveryPlanner):
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
                    self.bypass_paths[i][j]=sector_path(o['p'][i,active],int(np.flatnonzero(active==j)[0]),center,velocity,obstacles,radii)
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
