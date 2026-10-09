"""Public bearing coverage for the C209 handover, not a capture predicate."""
import numpy as np


def coverage_fraction(p, target):
    """Compact active team: one when no open half-plane contains all members.

    For largest circular angular gap g, min(1, (2*pi-g)/pi) measures
    how much of a semicircle the complementary bearing arc covers.
    A coincident target/pursuer bearing is undefined: return zero for that team.
    """
    delta=np.asarray(p,dtype=float)-np.asarray(target,dtype=float)[:,None]
    angles=np.sort(np.arctan2(delta[...,1],delta[...,0]),axis=-1)
    gaps=np.diff(np.concatenate([angles,angles[:,:1]+2*np.pi],axis=-1),axis=-1)
    result=np.clip((2*np.pi-gaps.max(-1))/np.pi,0.,1.)
    return np.where((np.linalg.norm(delta,axis=-1)<=1e-9).any(-1),0.,result)


def team_weight(o):
    radius=np.linalg.norm(o['p']-o['target'][:,None],axis=-1)
    distance=np.clip((radius.max(-1)-8.)/4.,0.,1.)
    return np.where(o['measured'],distance*coverage_fraction(o['p'],o['target']),0.)
