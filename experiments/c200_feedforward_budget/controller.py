"""Keep feasible target feedforward; scale the geometry correction uniformly.

This controls a proposal before the existing safety projection. It does not
assert that downstream obstacle avoidance preserves formation convergence.
"""
import numpy as np
from experiments.c145_feasibility_reset.planar import norm,unit,limit


def allocate(feed,correction,speed=4.):
    feed=limit(np.asarray(feed,float),speed)
    q=np.asarray(correction,float)
    a=np.sum(q*q,axis=-1);b=np.sum(feed*q,axis=-1)
    c=np.sum(feed*feed,axis=-1)-speed**2
    root=(-b+np.sqrt(np.maximum(b*b-a*c,0.)))/np.maximum(a,1e-12)
    alpha=np.where(a>1e-12,np.clip(root,0.,1.),1.)
    return feed+alpha[...,None]*q,alpha


def budget_polar(p,target,target_v,radius,c,extra):
    delta=p-target[...,None,:];r=norm(delta);er=unit(delta)
    et=np.stack([-er[...,1],er[...,0]],-1)
    angle=np.arctan2(delta[...,1],delta[...,0]);order=np.argsort(angle,axis=-1)
    sorted_angle=np.take_along_axis(angle,order,axis=-1)
    forward=(np.roll(sorted_angle,-1,axis=-1)-sorted_angle)%(2*np.pi)
    tang=np.take_along_axis(c.angular_gain*(forward-np.roll(forward,1,axis=-1)),np.argsort(order,axis=-1),axis=-1)*r
    q=-c.radial_gain*(r-np.asarray(radius)[...,None])[...,None]*er+tang[...,None]*et
    feed=target_v[...,None,:]+extra
    return allocate(feed,q,c.pursuer_speed)[0]
