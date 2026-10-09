"""Common action space, safety filter and sensing for all comparisons.

radius: fixed / algebraic disk margin / point-preview / disk-preview.
Preview uses a planar actuator surrogate and a fixed candidate grid.
"""
from dataclasses import dataclass,asdict
import numpy as np
from experiments.c145_feasibility_reset.planar import norm,unit,limit
from .geometry import disk_margin,gap_and_dist


@dataclass(frozen=True)
class Method:
    name:str='fixed35'
    kind:str='fixed'
    radius:float=3.5
    predictor:str='cv'
    uncertainty_linear:float=0.
    uncertainty_quadratic:float=1.25
    horizon:float=1.
    preview_dt:float=.2
    replan_steps:int=3
    response:bool=True
    radii:tuple=(3.2,3.5,3.8,4.1,4.4,4.7)
    lead_scales:tuple=(1.,)


def polar(p,v,target,target_v,radius,c,lead=1.):
    delta=p-target[...,None,:];r=norm(delta);er=unit(delta);et=np.stack([-er[...,1],er[...,0]],-1)
    a=np.arctan2(delta[...,1],delta[...,0]);order=np.argsort(a,axis=-1)
    sa=np.take_along_axis(a,order,axis=-1)
    forward=(np.roll(sa,-1,axis=-1)-sa)%(2*np.pi)
    tang=np.take_along_axis(c.angular_gain*(forward-np.roll(forward,1,axis=-1)),np.argsort(order,axis=-1),axis=-1)*r
    u=np.asarray(lead)[...,None,None]*target_v[...,None,:]-c.radial_gain*(r-np.asarray(radius)[...,None])[...,None]*er+tang[...,None]*et
    return limit(u,c.pursuer_speed)


class Controller:
    def __init__(self,n,method,cfg):
        self.method=method;self.cfg=cfg;self.calls=0
        self.radius=np.full(n,method.radius);self.lead=np.ones(n)
        self.last_margin=np.zeros(n);self.last_uncertainty=np.zeros(n)
        self.infeasible_steps=np.zeros(n,int)

    def uncertainty(self,age):
        m=self.method
        return m.uncertainty_linear*age+m.uncertainty_quadratic*age**2

    def command(self,o):
        m=self.method;c=self.cfg
        self.last_margin,_=disk_margin(o['p'],o['target'],c.target_buffer,c.radius,c.max_gap)
        self.last_uncertainty=self.uncertainty(o['age'])
        if m.kind=='algebraic':
            # Regular triangle: angular error tolerance = k R.
            k=.5-np.sqrt(3)/2/np.tan(c.max_gap)-np.sqrt(3)/2/np.sin(c.max_gap)
            need=self.uncertainty(o['age']+.3)
            lower=np.maximum(c.target_buffer+need,need/k)
            upper=c.radius-need
            self.radius=np.clip(np.maximum(m.radius,lower),3.2,4.7)
            self.infeasible_steps+=(lower>upper)
        elif m.kind in ('point_preview','disk_preview') and self.calls % m.replan_steps==0:
            self._choose(o)
        elif m.kind!='fixed' and m.kind not in ('point_preview','disk_preview'):
            raise ValueError(m.kind)
        self.calls+=1
        return polar(o['p'],o['v'],o['target'],o['target_v'],self.radius,c,self.lead)

    def _choose(self,o):
        m=self.method;c=self.cfg;n=len(o['p'])
        rs,leads=np.meshgrid(m.radii,m.lead_scales,indexing='ij');rs=rs.ravel();leads=leads.ravel();k=len(rs)
        p=np.repeat(o['p'][:,None],k,axis=1);v=np.repeat(o['v'][:,None],k,axis=1)
        target=np.repeat(o['target'][:,None],k,axis=1);tv=np.repeat(o['target_v'][:,None],k,axis=1)
        score=np.zeros((n,k));worst=np.full((n,k),np.inf)
        for step in range(round(m.horizon/m.preview_dt)):
            h=(step+1)*m.preview_dt
            # Extrapolation uses the same current information for every candidate.
            target= o['target'][:,None]+h*tv
            u=polar(p,v,target,tv,rs[None],c,leads[None])
            if m.response:
                # Integrate at the simulator's planar substep resolution.
                for _ in range(5):
                    dd=m.preview_dt/5
                    v=limit(v+dd*limit(c.velocity_kp*(u-v),c.acceleration),c.pursuer_speed)
                    p+=dd*v
            else:v=u;p+=m.preview_dt*v
            margin,_=disk_margin(p,target,c.target_buffer,c.radius,c.max_gap)
            if m.kind=='disk_preview':
                # The horizon has no promise of future sensing. During an observed
                # step use a single-frame uncertainty; during loss let it grow.
                age=np.where(o['measured'],.1,o['age']+h)
                eps=self.uncertainty(age)[:,None]
            else:eps=0.
            slack=margin-eps;worst=np.minimum(worst,slack)
            score+=12*np.minimum(slack,0.)**2+.3*np.sum((u-v)**2,axis=(-2,-1))
            if c.obstacles:
                clear=(norm(p[:,:,:,None]-o['obstacles'][:,None,None])-o['radii'][:,None,None]-c.obstacle_buffer).min((-1,-2))
                score+=100*np.minimum(clear-.5,0.)**2
            bound=(c.arena_half-1-np.abs(p)).min((-1,-2))
            score+=100*np.minimum(bound,0.)**2
        # Modest regularization keeps the no-loss controller near its tuned
        # baseline while allowing larger error tolerance when needed.
        score+=.10*(rs[None]-m.radius)**2+.15*(rs[None]-self.radius[:,None])**2
        chosen=np.argmin(score,axis=1)
        self.radius=rs[chosen];self.lead=leads[chosen]
        self.infeasible_steps+=(worst[np.arange(n),chosen]<0)
