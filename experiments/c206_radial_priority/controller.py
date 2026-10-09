"""Lexicographic radial/tangential projection of a nominal polar velocity.

Minimize radial command error first, then tangential error inside the 4 m/s
ball. This is a nominal velocity allocation, not a closed-loop certificate;
the inherited safety projection and bypass can change both components.
"""
import numpy as np
from experiments.c145_feasibility_reset.planar import norm,unit,limit


def radial_project(nominal,radial,speed=4.):
    if speed<=0:raise ValueError('speed must be positive')
    nominal=np.asarray(nominal,float);radial=np.asarray(radial,float)
    length=norm(radial);er=unit(radial);et=np.stack([-er[...,1],er[...,0]],-1)
    ur=np.clip(np.sum(nominal*er,axis=-1),-speed,speed)
    capacity=np.sqrt(np.maximum(speed**2-ur**2,0.))
    ut=np.clip(np.sum(nominal*et,axis=-1),-capacity,capacity)
    command=ur[...,None]*er+ut[...,None]*et
    return np.where((length>1e-9)[...,None],command,limit(nominal,speed))


def radial_priority_polar(p,target,target_v,radius,c,extra):
    delta=p-target[...,None,:];r=norm(delta);er=unit(delta)
    et=np.stack([-er[...,1],er[...,0]],-1)
    angle=np.arctan2(delta[...,1],delta[...,0]);order=np.argsort(angle,axis=-1)
    sorted_angle=np.take_along_axis(angle,order,axis=-1)
    forward=(np.roll(sorted_angle,-1,axis=-1)-sorted_angle)%(2*np.pi)
    tang=np.take_along_axis(c.angular_gain*(forward-np.roll(forward,1,axis=-1)),np.argsort(order,axis=-1),axis=-1)*r
    q=-c.radial_gain*(r-np.asarray(radius)[...,None])[...,None]*er+tang[...,None]*et
    feed=limit(target_v[...,None,:]+extra,c.pursuer_speed)
    return radial_project(feed+q,er,c.pursuer_speed)
