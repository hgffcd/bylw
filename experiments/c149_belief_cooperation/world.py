"""Independent noisy sensors and legal local memories; truth ends at the sensor.

Artificial sensor interruptions are a controlled diagnostic, not natural LOS.
The predictor consumes local, translation-invariant histories only. All messages
are shared ideally; this experiment makes no communication-limited claim.
"""
from dataclasses import dataclass, replace
import numpy as np
from experiments.c145_feasibility_reset.planar import visibility, norm, limit, unit
from experiments.c145_feasibility_reset.preview_evader import PreviewWorld
from experiments.c146_loss_geometry.world import Scenario
from experiments.c146_loss_geometry.controllers import polar
from experiments.c146_loss_geometry.belief import extrapolate


@dataclass(frozen=True)
class Conditions:
    loss: float = 2.
    start: float = 12.
    asymmetric: bool = True
    fov: float = 120.
    noise: bool = True


class LocalBelief:
    def __init__(self, sigma, model='ct'):
        self.sigma = sigma.copy(); self.model = model
        shape = sigma.shape
        self.p = np.zeros(shape+(2,)); self.v = self.p.copy()
        self.omega = np.zeros(shape); self.stamp = np.full(shape, -100.)
        self.valid = np.zeros(shape, bool); self.feature = np.zeros(shape+(12,), np.float32)

    def query(self, time):
        age = np.maximum(time-self.stamp, 0.)
        if self.model=='damped_ct':
            horizon=5.*(1-np.exp(-age/5.))
            p,v=extrapolate(self.p,self.v,self.omega,horizon,'ct')
            v*=np.exp(-age[...,None]/5.)
        else:p,v = extrapolate(self.p, self.v, self.omega, age, self.model)
        # Per-coordinate approximation; calibration is separate and empirical.
        variance = self.sigma**2 + (.75*self.sigma*age)**2 + (.30*age**2)**2 + (.12*age)**2
        return p,v,age,np.maximum(variance, .001)

    def update(self, time, mask, position, velocity):
        prior,_,_,_ = self.query(time)
        innovation = np.zeros_like(self.p)
        ids = np.where(mask)
        if len(ids[0]):
            p = position[ids]; v = velocity[ids]
            if not np.isfinite(p).all() or not np.isfinite(v).all():
                raise ValueError('nonfinite positive measurement')
            valid = self.valid[ids]; dt = time-self.stamp[ids]
            innovation[ids] = np.where(valid[:,None], p-prior[ids], 0.)
            # Reset after a gap; use a time-aware smoother while measurements recur.
            weight = np.where(valid, 1-np.exp(-dt/.08), 1.)
            newv = (1-weight[:,None])*self.v[ids] + weight[:,None]*v
            old = self.v[ids]
            angle = np.arctan2(old[:,0]*newv[:,1]-old[:,1]*newv[:,0], (old*newv).sum(-1))
            good = valid & (dt>1e-8) & (dt<.31) & (norm(old)>.5) & (norm(newv)>.5)
            turn = np.clip(angle/np.maximum(dt,1e-8), -.8, .8)
            self.omega[ids] = np.where(good, .6*self.omega[ids]+.4*turn, 0.)
            self.p[ids] = np.where(valid[:,None], (1-weight[:,None])*prior[ids]+weight[:,None]*p,p)
            self.v[ids] = newv; self.stamp[ids] = time; self.valid[ids] = True
        p,v,age,var = self.query(time)
        self.feature = np.concatenate([v/4, np.clip(innovation,-5,5)/2,
            self.omega[...,None], np.minimum(age,10)[...,None]/3,
            self.sigma[...,None], mask[...,None], self.valid[...,None],
            np.sqrt(var)[...,None]/5, self.v/4], -1).astype(np.float32)
        self.feature[~self.valid] = 0.
        return p,v,age,var


def fuse(p,v,var,age,valid,mode):
    if not valid.any(1).all():
        raise ValueError('an episode has no initial legal team measurement')
    if mode == 'precision': weight = valid/np.maximum(var,.001)
    elif mode == 'average': weight = valid.astype(float)
    elif mode == 'freshest':
        score = np.where(valid, age+1e-5*var, np.inf)
        weight = np.eye(3)[score.argmin(1)]
    else: raise ValueError(mode)
    weight /= weight.sum(1,keepdims=True)
    return (weight[...,None]*p).sum(1), (weight[...,None]*v).sum(1), weight


class LocalWorld(PreviewWorld):
    def __init__(self, seeds, conditions=Conditions(), fusion='precision', predictor='ct', learned=None):
        cfg = Scenario(sensor='visual',loss_duration=conditions.loss,loss_start=conditions.start,
                       fov_deg=conditions.fov).config()
        super().__init__(seeds, replace(cfg, desired_radius=3.8))
        self.conditions=conditions; self.fusion=fusion; self.learned=learned
        self.learned_hidden=None; self.tick=0; self.cached=None
        self.sigmas=np.empty((len(seeds),3)); self.phases=np.empty((len(seeds),3))
        self.noise=np.empty((300,len(seeds),3,4),np.float32)
        for i,seed in enumerate(seeds):
            rng=np.random.default_rng(np.random.SeedSequence([int(seed),149]))
            self.sigmas[i]=rng.permutation([.06,.15,.30]) if conditions.noise else .0001
            self.phases[i]=rng.permutation([0.,2.,4.])
            self.noise[:,i]=rng.standard_normal((300,3,4))
        self.local=LocalBelief(self.sigmas,predictor)
        self.natural_measured=np.ones(len(seeds),bool)

    def sensor(self):
        natural,*_=visibility(self.p,self.target,self.yaw,self.obstacles,self.radii,self.cfg)
        artificial=np.zeros_like(natural)
        if self.conditions.asymmetric and self.time>=5:
            artificial |= ((self.time-5+self.phases)%6)<2-1e-7
        if self.conditions.start<=self.time<self.conditions.start+self.conditions.loss-1e-7:
            artificial[:]=True
        mask=natural & ~artificial
        z=self.noise[min(self.tick,299)]
        position=self.target[:,None]+self.sigmas[...,None]*z[...,:2]
        velocity=self.tv[:,None]+.75*self.sigmas[...,None]*z[...,2:]
        return mask,np.where(mask[...,None],position,np.nan),np.where(mask[...,None],velocity,np.nan),natural,artificial

    def observe(self):
        if self.cached is not None and self.cached[0]==self.time:return self.cached[1]
        mask,position,velocity,natural,artificial=self.sensor()
        p,v,age,var=self.local.update(self.time,mask,position,velocity)
        base_p,base_v,base_var=p.copy(),v.copy(),var.copy()
        if self.learned is not None:
            import torch
            with torch.no_grad():
                x=torch.from_numpy(self.local.feature.reshape(-1,12))[:,None]
                mean,logvar,self.learned_hidden=self.learned(x,self.learned_hidden)
                correction=mean[:,0].numpy().reshape(len(self.seeds),3,4)*np.array([5,5,4,4])
                variance=np.exp(logvar[:,0].numpy()).reshape(len(self.seeds),3,4)*np.array([25,25,16,16])
            p=p+correction[...,:2];v=limit(v+correction[...,2:],3.2)
            var=variance[...,:2].mean(-1)*getattr(self.learned,'temperature',1.)
        self.estimate,self.estimate_v,weight=fuse(p,v,var,age,self.local.valid,self.fusion)
        self.age=np.min(np.where(self.local.valid,age,np.inf),1)
        self.visible=natural;self.measured=mask.any(1);self.natural_measured=natural.any(1)
        omega=(weight*self.local.omega).sum(1)
        o=dict(p=self.p.copy(),v=self.pv.copy(),target=self.estimate.copy(),target_v=self.estimate_v.copy(),
               age=self.age.copy(),measured=self.measured.copy(),omega=omega,
               obstacles=self.obstacles.copy(),radii=self.radii.copy(),
               local_p=p.copy(),local_v=v.copy(),local_var=var.copy(),local_age=age.copy(),
               local_valid=self.local.valid.copy(),local_mask=mask.copy(),
               prior_p=base_p,prior_v=base_v,prior_var=base_var,history_feature=self.local.feature.copy())
        # Evaluation-only fields remain outside observation / actor feature builders.
        self.last_natural=natural.copy();self.last_artificial=artificial.copy()
        self.cached=(self.time,o)
        return o

    def nominal(self,o):
        u=polar(o['p'],o['v'],o['target'],o['target_v'],3.8,self.cfg)
        accel=o['omega'][:,None]*np.stack([-o['target_v'][:,1],o['target_v'][:,0]],-1)
        return limit(u+.5*accel[:,None],self.cfg.pursuer_speed)

    def step(self,obs,residual=None):
        result=super().step(obs,residual)
        self.tick+=1
        return result


def action_residual(o, actions):
    er=unit(o['p']-o['target'][:,None]);et=np.stack([-er[...,1],er[...,0]],-1)
    radial=np.array([0,1,-1,0,0])[actions];tangent=np.array([0,0,0,1,-1])[actions]
    return .4*(radial[...,None]*er+tangent[...,None]*et)
