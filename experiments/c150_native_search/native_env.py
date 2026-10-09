"""Actual Isaac action/reward bridge; imported only after AppLauncher.

Synchronous finite-horizon rollouts are reset explicitly by the runner. No
automatic partial resets: PPO bootstraps zero at that registered horizon.
"""
import numpy as np
import torch
from experiments.c146_loss_geometry.native_env import PilotEnv
from experiments.c145_feasibility_reset.planar import norm,wrap
from bylw_isaac_p5.core.action_adapter import HighLevelCommand
from bylw_isaac_p5.core.low_level_controller import quaternion_to_euler_wxyz
from .task import geometry3
from .policy import features
from experiments.c171_variable_scale.geometry import active_geometry


class SearchEnv(PilotEnv):
    control_tick=0

    def sync_world(self):
        pos,vel=super().sync_world();w=self.pilot
        w.z=(pos-self.scene.env_origins[:,None])[...,2].detach().cpu().numpy().copy()
        w.vz=vel[...,2].detach().cpu().numpy().copy()
        w.tz=(self._evader.data.root_pos_w-self.scene.env_origins)[:,2].detach().cpu().numpy().copy()
        w.tvz=self._evader.data.root_lin_vel_w[:,2].detach().cpu().numpy().copy()
        return pos,vel

    def observe_now(self):
        self.sync_world();self.pilot.time=round(self.control_tick*.1,8)
        return self.pilot.observe()

    def policy_features(self):return features(self.pilot,self.observe_now())

    def _get_observations(self):
        agents=len(self.cfg.possible_agents)
        x=np.zeros((self.num_envs,agents,34),np.float32) if self.pilot is None else self.policy_features()
        return {a:torch.as_tensor(x[:,i],device=self.device) for i,a in enumerate(self.cfg.possible_agents)}

    def _get_states(self):
        if self.pilot is None:return torch.zeros(self.num_envs,self.cfg.state_space,device=self.device)
        features_now=self.policy_features()
        return torch.as_tensor(features_now.reshape(self.num_envs,-1),device=self.device)

    def _get_rewards(self):
        if self.pilot is None:r=np.zeros(self.num_envs)
        else:
            self.record_safety()
            o=self.observe_now();w=self.pilot
            if w.p.shape[1] == 3:
                g,d,gap=geometry3(w.p,w.z,w.target,w.tz)
            else:
                g,d,gap,_,_=active_geometry(w.p,w.z,w.target,w.tz,np.ones(w.p.shape[:2],dtype=bool))
            r=1.5*(g&o['measured'])+.1*o['measured']-.06*np.minimum(np.abs(d-6),20).mean(1)
            r-=.4*np.maximum(gap.max(1)-np.deg2rad(150),0)
            r-=.03*np.minimum(o['age'],10)+4*self.failures.any(1)
        self.last_reward=r.copy();t=torch.as_tensor(r,dtype=torch.float32,device=self.device)
        return {a:t for a in self.cfg.possible_agents}

    def _get_dones(self):
        if self.pilot is not None:self.record_safety()
        z={a:torch.zeros(self.num_envs,dtype=torch.bool,device=self.device) for a in self.cfg.possible_agents}
        return z,{a:v.clone() for a,v in z.items()}

    def _pre_physics_step(self,actions):
        o=self.observe_now();w=self.pilot
        aa=np.stack([actions[a].detach().cpu().numpy().reshape(self.num_envs) for a in self.cfg.possible_agents],1).astype(int)
        if ((aa<0)|(aa>=8)).any():raise ValueError('invalid categorical physical action')
        u,yaw_goal,z=w.commands(o,aa);ev=w.evader_command()
        self.last_obs=o;self.last_action=aa;self.last_command=u.copy();self.last_altitude=z.copy()
        self.last_prediction_error=norm(o['target']-w.target)
        yr=np.clip(2*wrap(yaw_goal-w.yaw),-1.5,1.5)
        def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)
        agents=len(self._pursuers)
        self._high_level_command=HighLevelCommand(tensor(u),tensor(z),tensor(yr),tensor(norm(u)),tensor(np.zeros((self.num_envs,agents,2))))
        ey=quaternion_to_euler_wxyz(self._evader.data.root_quat_w)[:,2].detach().cpu().numpy()
        eyr=np.clip(2*wrap(np.arctan2(ev[:,1],ev[:,0])-ey),-1.5,1.5)
        self._evader_high_level_command=HighLevelCommand(tensor(ev),tensor(np.full(self.num_envs,3.)),tensor(eyr),tensor(norm(ev)),tensor(np.zeros((self.num_envs,2))))
        self.measurement_loss+=~o['measured'];self.control_tick+=1

    def record_safety(self):
        if self.pilot is None:return
        self.sync_world();w=self.pilot
        active=np.asarray(w.active,dtype=bool)
        xyz=np.concatenate([w.p,w.z[...,None]],-1);target=np.concatenate([w.target,w.tz[:,None]],-1)
        td_all=norm(xyz-target[:,None]);td=np.where(active,td_all,np.inf).min(1)
        pair_distances=[np.where(active[:,i]&active[:,j],norm(xyz[:,i]-xyz[:,j]),np.inf)
                        for i in range(xyz.shape[1]) for j in range(i+1,xyz.shape[1])]
        pd=np.stack(pair_distances,1).min(1) if pair_distances else np.full(self.num_envs,np.inf)
        oc_all=norm(w.p[:,:,None]-w.obstacles[:,None])-w.radii[:,None]
        oc=np.where(active[:,:,None],oc_all,np.inf).min((1,2))
        toc=(norm(w.target[:,None]-w.obstacles)-w.radii).min(1)
        height=np.where(active,(w.z<2)|(w.z>4),False).any(1)|(np.abs(w.tz-3)>1)
        boundary=np.where(active[:,:,None],np.abs(w.p)>50,False).any((1,2))|(np.abs(w.target)>50).any(1)
        finite=np.isfinite(xyz).all((1,2))&np.isfinite(target).all(1)
        self.failures|=np.stack([td<1.5,pd<1,oc<.5,toc<.5,height,boundary,~finite],1)
        self.minima=np.minimum(self.minima,np.stack([td,pd,oc,toc],1))
        self.peak_speed=np.maximum(self.peak_speed,norm(w.pv).max(1))

    def freeze_inactive(self):
        """Stop failed bodies at the event pose before the next physics step.

        P5-C treats an exited vehicle as removed from the active team.  Zeroing
        its translational/angular velocity models that event without moving the
        body or changing any active member's state.
        """
        if self.pilot is None:
            return
        self.sync_world()
        active=np.asarray(self.pilot.active,dtype=bool)
        ids=torch.arange(self.num_envs,device=self.device)
        for i,robot in enumerate(self._pursuers):
            inactive=~active[:,i]
            if not inactive.any():
                continue
            # Isaac exposes inference tensors during rollout; state writes
            # must occur in a normal autograd context.
            with torch.inference_mode(False):
                linear=robot.data.root_lin_vel_w.detach().clone()
                angular=robot.data.root_ang_vel_w.detach().clone()
                mask=torch.as_tensor(inactive,device=self.device)
                linear[mask]=0.0
                angular[mask]=0.0
                robot.write_root_velocity_to_sim(torch.cat((linear,angular),-1),ids)

    def initialize_case(self,seeds):
        self.pilot=None;self.reset();self.control_tick=0
        w=self.world_factory(seeds);self.pilot=w;ids=torch.arange(self.num_envs,device=self.device)
        def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)
        for i,robot in enumerate(self._pursuers):
            self._write_vehicle_reset(robot,ids,tensor(np.concatenate([w.p[:,i],w.z[:,i,None]],1)),tensor(w.yaw[:,i]))
        self._write_vehicle_reset(self._evader,ids,tensor(np.concatenate([w.target,w.tz[:,None]],1)),tensor(w.heading))
        self._obstacle_xy_local[:]=tensor(w.obstacles);self._obstacle_radius[:]=tensor(w.radii)
        for j,robot in enumerate(self._obstacles):
            rs=robot.data.default_root_state.clone();rs[:,:2]=self.scene.env_origins[:,:2]+tensor(w.obstacles[:,j])
            rs[:,2]=4.;rs[:,3]=1;rs[:,4:]=0
            robot.write_root_pose_to_sim(rs[:,:7]);robot.write_root_velocity_to_sim(rs[:,7:])
        self.scene.write_data_to_sim();self.sim.forward();self.scene.update(self.physics_dt)
        self.minima=np.full((self.num_envs,4),np.inf);self.failures=np.zeros((self.num_envs,7),bool)
        self.peak_speed=np.zeros(self.num_envs);self.measurement_loss=np.zeros(self.num_envs,int)
        self._high_level_command=None;self._evader_high_level_command=None;self.record_safety()
        if not np.allclose(w.p,w.initial_p,atol=3e-4,rtol=1e-7):raise RuntimeError('physical reset mismatch')
        if not np.allclose(w.z,w.initial_z,atol=3e-4,rtol=1e-7):raise RuntimeError('physical height reset mismatch')
        self.observe_now()
