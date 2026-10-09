"""C210 public relative radial velocity damping before formation saturation."""
import numpy as np
from experiments.c145_feasibility_reset.planar import norm,unit,limit


def damping_delta(o,c,anticipation):
    delta=o['p']-o['target'][:,None];er=unit(delta)
    near=np.clip((12.-norm(delta).max(-1))/4.,0.,1.)
    enabled=o['measured'] & (o['searcher']<0)
    gain=c.radial_gain*np.asarray(anticipation)*np.where(enabled,near,0.)
    radial_velocity=np.sum((o['v']-o['target_v'][:,None])*er,axis=-1)
    return -gain[:,None,None]*radial_velocity[...,None]*er


def damped_polar(o,radius,c,anticipation):
    # The original polar field, with one additive term BEFORE its first limit.
    p=o['p'];target=o['target'];target_v=o['target_v']
    delta=p-target[...,None,:];r=norm(delta);er=unit(delta);et=np.stack([-er[...,1],er[...,0]],-1)
    a=np.arctan2(delta[...,1],delta[...,0]);order=np.argsort(a,axis=-1)
    sa=np.take_along_axis(a,order,axis=-1)
    forward=(np.roll(sa,-1,axis=-1)-sa)%(2*np.pi)
    tang=np.take_along_axis(c.angular_gain*(forward-np.roll(forward,1,axis=-1)),np.argsort(order,axis=-1),axis=-1)*r
    u=target_v[...,None,:]-c.radial_gain*(r-np.asarray(radius)[...,None])[...,None]*er+tang[...,None]*et
    return limit(u+damping_delta(o,c,anticipation),c.pursuer_speed)
