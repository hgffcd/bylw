"""C187: C186 v2 with separate transit and endpoint obstacle reserves.

C186 source is preserved unchanged for replay. V2 also follows intermediate
waypoints at the existing speed cap rather than braking at each polygon vertex.

No judge relaxation: 4..8 m radii, <=2 m spread, 90..150 degree gaps.
This controller uses public map/shared estimates only; it is not a certificate.
"""
import numpy as np
from scipy.optimize import minimize

from experiments.c184_route_recovery.planner import OFFSETS, VisibilityRouter, RecoveryPlanner


def points(x):
    angles=x[0]+np.deg2rad([0.,x[1],x[1]+x[2]])
    return x[3:,None]*np.c_[np.cos(angles),np.sin(angles)]


def parameters(offset):
    angle=np.arctan2(offset[:,1],offset[:,0])
    gaps=np.rad2deg((np.roll(angle,-1)-angle)%(2*np.pi))
    return np.r_[angle[0],gaps[:2],np.linalg.norm(offset,axis=-1)]


def refine_formation(p,center,obstacles,radii,n_static,previous=None):
    p=np.asarray(p);center=np.asarray(center);obs=np.asarray(obstacles);radii=np.asarray(radii)
    # Retired bodies retain the original 1.6 m projection reserve, plus .1 m.
    inflation=np.r_[np.full(n_static,.8),np.full(len(obs)-n_static,1.1)]
    envelopes=radii+inflation
    order=np.argsort(np.arctan2((p-center)[:,1],(p-center)[:,0]))
    slots=center+OFFSETS
    clear=(np.linalg.norm(slots[...,None,:]-obs,axis=-1)-envelopes).min((-1,-2))
    distance=np.linalg.norm(slots-p[order],axis=-1).sum(-1)
    priority=clear-.015*distance
    starts=[]
    if previous is not None:
        starts.append(parameters(previous[order]))
    chosen=[]
    for k in np.argsort(-priority):
        if all(np.linalg.norm(OFFSETS[k]-OFFSETS[j])>3 for j in chosen):
            chosen.append(int(k));starts.append(parameters(OFFSETS[k]))
            if len(chosen)>=8:break

    def constraints(x):
        goals=center+points(x);rr=x[3:]
        spread=np.array([rr[0]-rr[1],rr[0]-rr[2],rr[1]-rr[2]])
        return np.r_[x[1]+x[2]-212.,268.-x[1]-x[2],
                     1.8-spread,1.8+spread,
                     (np.linalg.norm(goals[:,None]-obs,axis=-1)-envelopes).ravel(),
                     (48.5-np.abs(goals)).ravel()]

    def objective(x):
        offset=points(x)
        value=np.sqrt(np.sum((center+offset-p[order])**2,axis=-1)+.04).sum()
        if previous is not None:value+=.8*np.sqrt(np.sum((offset-previous[order])**2,axis=-1)+.04).sum()
        return value

    bounds=[(-np.pi,np.pi),(92.,148.),(92.,148.)]+[(4.2,7.8)]*3
    candidates=[]
    for x0 in starts:
        x0=np.clip(x0,np.array(bounds)[:,0],np.array(bounds)[:,1])
        # Retain a currently feasible warm start even if a local solve fails.
        if constraints(x0).min()>=-1e-5:
            offset=points(x0)
            if all(np.linalg.norm(offset-other)>1. for other in candidates):candidates.append(offset)
        opt=minimize(objective,x0,method='SLSQP',bounds=bounds,
                     constraints={'type':'ineq','fun':constraints},
                     options={'maxiter':60,'ftol':1e-7})
        x=opt.x
        if not np.isfinite(x).all() or constraints(x).min() < -1e-5:continue
        offset=points(x)
        if all(np.linalg.norm(offset-other)>1. for other in candidates):candidates.append(offset)
    if not candidates:return None
    candidates=np.asarray(candidates)
    goals=center+candidates
    transit_envelopes=envelopes.copy()
    transit_envelopes[:n_static]=radii[:n_static]+.65
    router=VisibilityRouter(np.r_[obs,center[None]],np.r_[transit_envelopes,3.8],goals.reshape(-1,2))
    routes=[router.paths_from(start) for start in p[order]]
    travel=np.stack([r[0].reshape(-1,3)[:,i] for i,r in enumerate(routes)],axis=1)
    cost=travel.sum(-1)
    if previous is not None:cost+=.8*np.linalg.norm(candidates-previous[order],axis=-1).sum(-1)
    if not np.isfinite(cost).any():return None
    best=int(np.argmin(cost));paths=[None]*3;offset=np.zeros((3,2))
    for slot,agent in enumerate(order):
        paths[agent]=routes[slot][1][3*best+slot]
        offset[agent]=candidates[best,slot]
    return dict(paths=paths,offsets=offset,cost=float(cost[best]))


class TransitRecoveryPlanner(RecoveryPlanner):
    def __init__(self,batch,agents):
        super().__init__(batch,agents)
        self.precision_on=np.zeros(batch,bool)
        self.precision_changed=np.zeros(batch,bool)
        self.precision_entries=np.zeros(batch,int)
        self.precision_calls=np.zeros(batch,int)
        self.precision_no_plan=np.zeros(batch,int)
        self.blocked_since=np.full(batch,np.nan)
        self.precision_paths=[None]*batch
        self.precision_offsets=[None]*batch
        self.precision_next=np.zeros(batch)
        self.precision_clock=None
        self.precision_output=None
        self.precision_center=np.full((batch,2),np.nan)

    def propose(self,o,base,time):
        if self.precision_clock==time and self.precision_output is not None:return self.precision_output
        dt=0. if self.precision_clock is None else max(0.,time-self.precision_clock)
        self.precision_clock=time
        result,changed=super().propose(o,base,time)
        out=result.copy();changed=changed.copy();self.precision_changed[:]=False
        for i in range(len(base)):
            ids=np.flatnonzero(o['active'][i])
            eligible=(len(ids)==3 and bool(o['measured'][i]) and np.linalg.norm(o['target_v'][i])<.5)
            if not eligible:
                self.precision_on[i]=False;self.blocked_since[i]=np.nan
                self.precision_paths[i]=None;self.precision_offsets[i]=None
                self.precision_center[i]=np.nan
                continue
            if not np.isfinite(self.precision_center[i]).all():
                self.precision_center[i]=o['target'][i]
            else:
                predicted=self.precision_center[i]+dt*o['target_v'][i]
                self.precision_center[i]=predicted+(1.-np.exp(-dt/.7))*(o['target'][i]-predicted)
            blocked=self.on[i] and self.paths[i] is None
            if blocked and not np.isfinite(self.blocked_since[i]):self.blocked_since[i]=time
            if not blocked:self.blocked_since[i]=np.nan
            if not self.precision_on[i] and blocked and time-self.blocked_since[i]>=1.8:
                self.precision_on[i]=True;self.precision_entries[i]+=1;self.precision_next[i]=time
            if not self.precision_on[i]:continue
            p=o['p'][i,ids];center=self.precision_center[i]
            if time>=self.precision_next[i]-1e-6:
                self.precision_calls[i]+=1
                retired=o['p'][i,~o['active'][i]]
                obs=np.r_[o['obstacles'][i],retired]
                radii=np.r_[o['radii'][i],np.full(len(retired),.6)]
                plan=refine_formation(p,center,obs,radii,len(o['obstacles'][i]),self.precision_offsets[i])
                self.precision_next[i]=time+1.
                if plan is None:
                    self.precision_no_plan[i]+=1;self.precision_paths[i]=None
                else:
                    self.precision_paths[i]=plan['paths'];self.precision_offsets[i]=plan['offsets']
            if self.precision_paths[i] is None:continue
            changed[i]=True;self.precision_changed[i]=True
            for j,agent in enumerate(ids):
                path=self.precision_paths[i][j]
                while len(path)>1 and np.linalg.norm(path[0]-p[j])<.3:path=path[1:]
                self.precision_paths[i][j]=path
                goal=center+self.precision_offsets[i][j] if len(path)==1 else path[0]
                vel=.8*(goal-p[j])+o['target_v'][i]
                if len(path)>1:
                    direction=goal-p[j]
                    vel=1.2*direction/max(np.linalg.norm(direction),1e-9)+o['target_v'][i]
                out[i,agent]=vel*min(1.,1.2/max(np.linalg.norm(vel),1e-9))
        self.precision_output=(out,changed)
        return self.precision_output
