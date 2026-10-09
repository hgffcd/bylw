"""Distance-scaled angular gradient with the existing feasible-feed allocator.

V = kr/2 sum (r-R)^2 + kt R^2/2 sum (gap-2pi/N)^2.
For distinct bearings and nonzero r, q is -grad_p V. The angular component
therefore scales as R^2/r, unlike the inherited angular-rate controller's r.
With exact constant target feed, no compensation or safety/route projection,
the allocated ideal single-integrator velocity satisfies Vdot <= 0. This
does not certify the native closed loop, visibility, obstacles or capture.
"""
import numpy as np
from experiments.c145_feasibility_reset.planar import norm,unit
from experiments.c200_feedforward_budget.controller import allocate


def metric_correction(p,target,radius,c):
    delta=p-target[...,None,:];r=norm(delta);er=unit(delta)
    et=np.stack([-er[...,1],er[...,0]],-1)
    angle=np.arctan2(delta[...,1],delta[...,0]);order=np.argsort(angle,axis=-1)
    a=np.take_along_axis(angle,order,axis=-1)
    gaps=(np.roll(a,-1,axis=-1)-a)%(2*np.pi)
    imbalance=np.take_along_axis(gaps-np.roll(gaps,1,axis=-1),np.argsort(order,axis=-1),axis=-1)
    reference=np.asarray(radius)[...,None]
    tangent=c.angular_gain*imbalance*reference**2/np.maximum(r,1e-9)
    return -c.radial_gain*(r-reference)[...,None]*er+tangent[...,None]*et


def metric_budget_polar(p,target,target_v,radius,c,extra):
    q=metric_correction(p,target,radius,c)
    return allocate(target_v[...,None,:]+extra,q,c.pursuer_speed)[0]


def formation_potential(p,target,radius,c):
    d=p-target[...,None,:];r=norm(d);angle=np.sort(np.arctan2(d[...,1],d[...,0]),axis=-1)
    gaps=(np.roll(angle,-1,axis=-1)-angle)%(2*np.pi)
    reference=np.asarray(radius)
    return (.5*c.radial_gain*np.sum((r-reference[...,None])**2,axis=-1)
            +.5*c.angular_gain*reference**2*np.sum((gaps-2*np.pi/p.shape[-2])**2,axis=-1))
