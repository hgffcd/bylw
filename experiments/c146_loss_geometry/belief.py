"""Predictor state receives only positive measurements and a mask.

Truth is intentionally absent from the query/predict API. Constant-turn
extrapolation is a classical baseline, not a proposed contribution.
"""
import numpy as np


def extrapolate(position, velocity, omega, age, model='cv'):
    age=np.asarray(age)
    if model=='cv':return position+age[...,None]*velocity,velocity.copy()
    if model!='ct':raise ValueError(model)
    angle=omega*age
    sin_over_w=age*np.sinc(angle/np.pi)
    one_minus_cos_over_w=.5*omega*age**2*np.sinc(angle/(2*np.pi))**2
    orth=np.stack([-velocity[...,1],velocity[...,0]],-1)
    mean=position+sin_over_w[...,None]*velocity+one_minus_cos_over_w[...,None]*orth
    vel=np.cos(angle)[...,None]*velocity+np.sin(angle)[...,None]*orth
    return mean,vel


class Belief:
    def __init__(self,n,model='cv',turn_bound=.8):
        self.model=model;self.turn_bound=turn_bound
        self.position=np.zeros((n,2));self.velocity=np.zeros((n,2))
        self.omega=np.zeros(n);self.timestamp=np.full(n,np.nan)
        self.initialized=np.zeros(n,bool)

    def update(self,time,measured,position,velocity):
        measured=np.asarray(measured,dtype=bool)
        # Caller may use NaN placeholders for unseen states; they never enter memory.
        ids=np.flatnonzero(measured)
        if not len(ids):return
        p=np.asarray(position)[ids];v=np.asarray(velocity)[ids]
        if not np.isfinite(p).all() or not np.isfinite(v).all():raise ValueError('nonfinite measurement')
        old=self.velocity[ids];dt=time-self.timestamp[ids]
        cross=old[:,0]*v[:,1]-old[:,1]*v[:,0]
        dot=np.sum(old*v,-1)
        available=self.initialized[ids]&(dt>1e-8)&(dt<.151)&(np.linalg.norm(old,axis=-1)>.5)&(np.linalg.norm(v,axis=-1)>.5)
        observed=np.clip(np.arctan2(cross,dot)/np.maximum(np.nan_to_num(dt,nan=1),1e-8),-self.turn_bound,self.turn_bound)
        self.omega[ids]=np.where(available,.5*self.omega[ids]+.5*observed,0.)
        self.position[ids]=p;self.velocity[ids]=v
        self.timestamp[ids]=time;self.initialized[ids]=True

    def query(self,time):
        if not self.initialized.all():raise ValueError('each episode requires an initial legal observation')
        age=np.maximum(time-self.timestamp,0.)
        p,v=extrapolate(self.position,self.velocity,self.omega,age,self.model)
        return p,v,age
