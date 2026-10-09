"""Independent C146 layer; C145 sources and artifacts stay untouched."""
from dataclasses import dataclass
import numpy as np
from experiments.c145_feasibility_reset.planar import Config,visibility
from experiments.c145_feasibility_reset.preview_evader import PreviewWorld
from .belief import Belief
from .controllers import Controller


@dataclass(frozen=True)
class Scenario:
    name:str='pulse08'
    sensor:str='shared'
    loss_duration:float=.8
    loss_start:float=12.
    fov_deg:float=120.
    yaw_rate:float=1.5
    obstacles:int=4
    evader_ratio:float=.8
    turn_rate:float=.8
    layout:str='ring'
    duration:float=30.

    def config(self):
        return Config(sensor=self.sensor,blackout_start=self.loss_start,blackout_duration=self.loss_duration,
                      fov_half=np.deg2rad(self.fov_deg/2),yaw_rate=self.yaw_rate,obstacles=self.obstacles,
                      obstacle_buffer=.5,evader_ratio=self.evader_ratio,target_turn_rate=self.turn_rate,
                      layout=self.layout,duration=self.duration,desired_radius=3.5)


class LossWorld(PreviewWorld):
    def __init__(self,seeds,scenario,method):
        super().__init__(seeds,scenario.config())
        self.scenario=scenario;self.method=method
        self.belief=Belief(len(seeds),method.predictor,self.cfg.target_turn_rate)
        self.policy=Controller(len(seeds),method,self.cfg)
        self.natural_measured=np.ones(len(seeds),bool)
        self.sensor_history=[]

    def observe(self):
        c=self.cfg
        self.visible,*_=visibility(self.p,self.target,self.yaw,self.obstacles,self.radii,c)
        self.natural_measured=self.visible.any(1)
        measured=self.natural_measured.copy() if c.sensor=='visual' else np.ones(len(self.seeds),bool)
        if c.blackout_start<=self.time<c.blackout_start+c.blackout_duration-1e-7:measured[:]=False
        # Unobserved truth is discarded before crossing the belief boundary.
        p=np.where(measured[:,None],self.target,np.nan)
        v=np.where(measured[:,None],self.tv,np.nan)
        self.belief.update(self.time,measured,p,v)
        self.estimate,self.estimate_v,self.age=self.belief.query(self.time)
        self.measured=measured
        return dict(p=self.p.copy(),v=self.pv.copy(),target=self.estimate.copy(),target_v=self.estimate_v.copy(),
                    age=self.age.copy(),measured=measured.copy(),obstacles=self.obstacles.copy(),radii=self.radii.copy())

    def nominal(self,obs):return self.policy.command(obs)
