"""Exact centered-disk tolerance for a nondegenerate triangular enclosure.

The target must remain inside the triangle, at distances [r_min, r_max],
with each cyclic bearing gap <= theta, pi/2 < theta < pi. Each forbidden
large-angle segment is a disk on the exterior side of a triangle edge.
This is an elementary geometry construction, not a novelty claim.
"""
import numpy as np


def disk_margin(p, center, r_min=1.5, r_max=5.5, theta=np.deg2rad(150)):
    p=np.asarray(p);center=np.asarray(center)
    if not np.pi/2 < theta < np.pi:raise ValueError('requires obtuse gap threshold')
    d=p-center[...,None,:]
    angles=np.arctan2(d[...,1],d[...,0])
    order=np.argsort(angles,axis=-1)
    s=np.take_along_axis(p,order[...,None],axis=-2)
    edge=np.roll(s,-1,axis=-2)-s
    length=np.linalg.norm(edge,axis=-1)
    inward=np.stack([-edge[...,1],edge[...,0]],-1)/np.maximum(length[...,None],1e-12)
    midpoint=(s+np.roll(s,-1,axis=-2))/2
    signed=np.sum((center[...,None,:]-midpoint)*inward,axis=-1)
    arc_center=midpoint+inward*(length/2/np.tan(theta))[...,None]
    arc_radius=length/2/np.sin(theta)
    angular=np.linalg.norm(center[...,None,:]-arc_center,axis=-1)-arc_radius
    distance=np.linalg.norm(d,axis=-1)
    pieces=np.stack([(distance-r_min).min(-1),(r_max-distance).min(-1),
                     signed.min(-1),angular.min(-1)],-1)
    margin=pieces.min(-1)
    # Degenerate/coincident pursuers cannot enclose a disk of positive radius.
    area=np.abs(edge[...,0,0]*edge[...,1,1]-edge[...,0,1]*edge[...,1,0])
    margin=np.where((length.min(-1)>1e-8)&(area>1e-8),margin,-1.)
    return margin,pieces


def gap_and_dist(p, center):
    d=np.asarray(p)-np.asarray(center)[...,None,:]
    a=np.sort(np.arctan2(d[...,1],d[...,0]),axis=-1)
    gaps=np.diff(np.concatenate([a,a[...,:1]+2*np.pi],axis=-1),axis=-1)
    return gaps.max(-1),np.linalg.norm(d,axis=-1)
