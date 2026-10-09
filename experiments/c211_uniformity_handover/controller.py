"""Uniformity moderates C209's public angular-coverage handover."""
import numpy as np
from experiments.c209_coverage_handover.controller import team_weight as coverage_weight


def uniformity_fraction(p,target):
    """1/(1+E), E=max absolute gap error divided by the nominal 2*pi/N.

    This is a dimensionless handover heuristic, not a capture predicate or a
    stability certificate. Nonuniform bearings retain some formation control.
    The positive factor avoids a new hard mode switch or timer.
    """
    delta=np.asarray(p,dtype=float)-np.asarray(target,dtype=float)[:,None]
    angles=np.sort(np.arctan2(delta[...,1],delta[...,0]),axis=-1)
    gaps=np.diff(np.concatenate([angles,angles[:,:1]+2*np.pi],axis=-1),axis=-1)
    nominal=2*np.pi/delta.shape[1]
    error=np.max(np.abs(gaps/nominal-1.),axis=-1)
    factor=1./(1.+error)
    return np.where((np.linalg.norm(delta,axis=-1)<=1e-9).any(-1),0.,factor)


def team_weight(o):
    return coverage_weight(o)*uniformity_fraction(o['p'],o['target'])
