"""Original C175 projections with speed-dependent extra obstacle reserve.

Only the explicit low-speed recovery arm invokes this. The physical safety
judge and downstream measured-closing-speed response guard stay unchanged.
"""
import numpy as np
from experiments.c145_feasibility_reset.planar import norm,unit,limit


def obstacle_extra_reserve(closing):
    closing=np.maximum(closing,0.)
    return np.minimum(.8,.15+.35*closing+closing*closing/4.)


def precision_filter(view,command,o):
    c=view.cfg;u=command.copy();p=o['p']
    for _ in range(8):
        d=p-o['target'][:,None];normal=unit(d)
        reserve=2.+.4*np.minimum(o['age'],4.)+.5*np.sqrt(np.maximum(o['variance'],0.))
        lower=-2.4*(norm(d)-(c.target_buffer+reserve[:,None]))
        u+=np.maximum(lower-(normal*(u-o['target_v'][:,None])).sum(-1),0)[...,None]*normal
        for i in range(p.shape[1]):
            for j in range(i+1,p.shape[1]):
                d=p[:,i]-p[:,j];normal=unit(d)
                lower=-1.2*(norm(d)-(c.pair_buffer+.6))
                correction=.5*np.maximum(lower-(normal*(u[:,i]-u[:,j])).sum(-1),0)[:,None]*normal
                u[:,i]+=correction;u[:,j]-=correction
        for j in range(o['obstacles'].shape[1]):
            d=p-o['obstacles'][:,None,j];normal=unit(d)
            closing=np.maximum(-(normal*o['v']).sum(-1),0.)
            extra=obstacle_extra_reserve(closing)
            lower=-1.2*(norm(d)-(o['radii'][:,None,j]+c.obstacle_buffer+extra))
            u+=np.maximum(lower-(normal*u).sum(-1),0)[...,None]*normal
        for j in range(view.departed_p.shape[1]):
            d=p-view.departed_p[:,None,j];normal=unit(d)
            lower=-1.2*(norm(d)-1.6)
            relative=(normal*(u-view.departed_v[:,None,j])).sum(-1)
            u+=np.maximum(lower-relative,0)[...,None]*normal
        u=np.minimum(u,1.2*(c.arena_half-1-p))
        u=np.maximum(u,-1.2*(c.arena_half-1+p))
        u=limit(u,c.pursuer_speed)
    return u
