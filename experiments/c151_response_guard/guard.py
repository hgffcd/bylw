"""Measured-closing-speed braking reference filter; an empirical safeguard.

Not a control-barrier certificate: actual attitude dynamics, moving targets,
inconsistent constraints and finite projection passes can still cause failure.
"""
import numpy as np
from experiments.c145_feasibility_reset.planar import norm,unit,limit


def closing_limit(clearance,closing,accel=2.,delay=.35,gain=2.):
    admissible=np.sqrt((accel*delay)**2+2*accel*np.maximum(clearance,0))-accel*delay
    return -admissible+gain*np.maximum(closing-admissible,0)+2*np.maximum(-clearance,0)


def response_guard(command,o,speed=4.,passes=8,target_buffer=1.5):
    p=o['p'];v=o['v'];u=command.copy()
    active=np.asarray(o.get('active',np.ones(p.shape[:2],dtype=bool)),dtype=bool)
    for _ in range(passes):
        # Moving target: only shared estimate, never hidden target state.
        d=p-o['target'][:,None];n=unit(d)
        relative_v=v-o['target_v'][:,None];closing=np.maximum(-(n*relative_v).sum(-1),0)
        lower=closing_limit(norm(d)-target_buffer-.25,closing)
        actual=(n*(u-o['target_v'][:,None])).sum(-1)
        u+=np.maximum(lower-actual,0)[...,None]*n*active[...,None]
        for i in range(p.shape[1]):
            for j in range(i+1,p.shape[1]):
                pair_active=active[:,i]&active[:,j]
                if not pair_active.any():
                    continue
                d=p[:,i]-p[:,j];n=unit(d);closing=np.maximum(-(n*(v[:,i]-v[:,j])).sum(-1),0)
                lower=closing_limit(norm(d)-1.-.15,closing,accel=4.)
                violation=np.maximum(lower-(n*(u[:,i]-u[:,j])).sum(-1),0)
                violation*=pair_active
                u[:,i]+=.5*violation[:,None]*n;u[:,j]-=.5*violation[:,None]*n
        for j in range(o['obstacles'].shape[1]):
            d=p-o['obstacles'][:,None,j];n=unit(d);closing=np.maximum(-(n*v).sum(-1),0)
            lower=closing_limit(norm(d)-o['radii'][:,None,j]-.5-.15,closing)
            u+=np.maximum(lower-(n*u).sum(-1),0)[...,None]*n*active[...,None]
        for axis in range(2):
            for sign in [-1.,1.]:
                outward=sign*v[...,axis];closing=np.maximum(-outward,0)
                lower=closing_limit(49.5+sign*p[...,axis],closing)
                u[...,axis]+=sign*np.maximum(lower-sign*u[...,axis],0)
        u=limit(u,speed)
        u=np.where(active[...,None],u,0.)
    return u
