"""C146 adapter derived from C145 PilotEnv; imported only after AppLauncher.
Upstream file SHA256: a6257dc40e30ff773609933a3981ee52df0cfdde5a32b72c4be621af2377c648
Changes: injectable strict-observation world, pre-step diagnostic cache.
No changes to the C145 implementation or the underlying quadrotor assets.
"""
import numpy as np
import torch
from experiments.c145_feasibility_reset.planar import norm,wrap
from bylw_isaac_p5.core.action_adapter import HighLevelCommand
from bylw_isaac_p5.core.low_level_controller import quaternion_to_euler_wxyz
from bylw_isaac_p5.tasks.direct.bylw_pursuit.bylw_pursuit_env import BylwPursuitEnv

class PilotEnv(BylwPursuitEnv):
    pilot=None

    def _get_observations(self):
        return {a:torch.zeros(self.num_envs,self.cfg.observation_spaces[a],device=self.device) for a in self.cfg.possible_agents}

    def _get_states(self):return torch.zeros(self.num_envs,self.cfg.state_space,device=self.device)

    def _get_rewards(self):
        return {a:torch.zeros(self.num_envs,device=self.device) for a in self.cfg.possible_agents}

    def _get_dones(self):
        zero={a:torch.zeros(self.num_envs,dtype=torch.bool,device=self.device) for a in self.cfg.possible_agents}
        return zero,zero.copy()

    def sync_world(self):
        w=self.pilot
        pos,quat,vel,_=self._stack_pursuer_state()
        w.p=(pos-self.scene.env_origins[:,None])[...,:2].detach().cpu().numpy().copy()
        w.pv=vel[...,:2].detach().cpu().numpy().copy()
        w.yaw=quaternion_to_euler_wxyz(quat)[...,2].detach().cpu().numpy().copy()
        w.target=(self._evader.data.root_pos_w-self.scene.env_origins)[:,:2].detach().cpu().numpy().copy()
        w.tv=self._evader.data.root_lin_vel_w[:,:2].detach().cpu().numpy().copy()
        return pos,vel

    def _pre_physics_step(self, actions):
        self.sync_world();w=self.pilot
        obs=w.observe();u=w.safety_filter(w.nominal(obs),obs);ev=w.evader_command()
        self.last_obs=obs
        self.last_prediction_error=norm(obs['target']-w.target)
        yaw_goal=np.arctan2((obs['target'][:,None]-w.p)[...,1],(obs['target'][:,None]-w.p)[...,0])
        yr=np.clip(2*wrap(yaw_goal-w.yaw),-w.cfg.yaw_rate,w.cfg.yaw_rate)
        def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=self.device)
        agents=len(self._pursuers)
        self._high_level_command=HighLevelCommand(tensor(u),tensor(np.full((self.num_envs,agents),3.)),tensor(yr),tensor(norm(u)),tensor(np.zeros((self.num_envs,agents,2))))
        ey=quaternion_to_euler_wxyz(self._evader.data.root_quat_w)[:,2].detach().cpu().numpy()
        eyr=np.clip(2*wrap(np.arctan2(ev[:,1],ev[:,0])-ey),-1.5,1.5)
        self._evader_high_level_command=HighLevelCommand(tensor(ev),tensor(np.full(self.num_envs,3.)),tensor(eyr),tensor(norm(ev)),tensor(np.zeros((self.num_envs,2))))
        self.measurement_loss+=~obs['measured']

    def record_safety(self):
        if self.pilot is None:return
        w=self.pilot;c=w.cfg
        pos,vel=self.sync_world()
        target_min=norm(w.p-w.target[:,None]).min(1)
        pair_distances=[norm(w.p[:,i]-w.p[:,j]) for i in range(w.p.shape[1]) for j in range(i+1,w.p.shape[1])]
        pair_min=np.stack(pair_distances,1).min(1) if pair_distances else np.full(len(w.seeds),np.inf)
        clearance=(norm(w.p[:,:,None]-w.obstacles[:,None])-w.radii[:,None]).min((1,2))
        target_clearance=(norm(w.target[:,None]-w.obstacles)-w.radii).min(1)
        z=pos[...,2].detach().cpu().numpy();ez=self._evader.data.root_pos_w[:,2].detach().cpu().numpy()
        height=(np.abs(z-3.)>1.).any(1)|(np.abs(ez-3.)>1.)
        boundary=(np.abs(w.p)>50).any((1,2))|(np.abs(w.target)>50).any(1)
        finite=np.isfinite(w.p).all((1,2))&np.isfinite(w.target).all(1)&np.isfinite(z).all(1)&np.isfinite(ez)
        self.minima=np.minimum(self.minima,np.stack([target_min,pair_min,clearance,target_clearance],1))
        self.failures|=np.stack([target_min<1.5,pair_min<1.,clearance<.5,target_clearance<.5,height,boundary,~finite],1)
        self.peak_speed=np.maximum(self.peak_speed,norm(w.pv).max(1))

    def _apply_action(self):
        self.record_safety()
        super()._apply_action()

    def initialize_case(self,cfg,seeds,preview=False):
        # Base reset first; then author the explicit new-task matched geometry.
        self.pilot=None;self.reset()
        w=self.world_factory(seeds);w.radii.fill(1.2);self.pilot=w
        ids=torch.arange(self.num_envs,device=self.device)
        def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=self.device)
        z=np.full((self.num_envs,1),3.)
        for i,robot in enumerate(self._pursuers):
            self._write_vehicle_reset(robot,ids,tensor(np.concatenate([w.p[:,i],z],1)),tensor(w.yaw[:,i]))
        self._write_vehicle_reset(self._evader,ids,tensor(np.concatenate([w.target,z],1)),tensor(w.heading))
        self._obstacle_xy_local[:]=tensor(w.obstacles);self._obstacle_radius[:]=1.2
        for j,robot in enumerate(self._obstacles):
            rs=robot.data.default_root_state.clone();rs[:,:2]=self.scene.env_origins[:,:2]+tensor(w.obstacles[:,j])
            rs[:,2]=self.cfg.obstacle_height_m/2;rs[:,3]=1.;rs[:,4:]=0.
            robot.write_root_pose_to_sim(rs[:,:7]);robot.write_root_velocity_to_sim(rs[:,7:])
        self.scene.write_data_to_sim();self.sim.forward();self.scene.update(self.physics_dt)
        self.minima=np.full((self.num_envs,4),np.inf);self.failures=np.zeros((self.num_envs,7),bool)
        self.peak_speed=np.zeros(self.num_envs);self.measurement_loss=np.zeros(self.num_envs,int)
        self._high_level_command=None;self._evader_high_level_command=None
        self.record_safety()
        if not np.allclose(w.p,w.initial_p,atol=3e-4,rtol=1e-7):
            raise RuntimeError('physical pursuer reset differs from registered geometry')
        if not np.allclose(w.target,w.initial_target,atol=3e-4,rtol=1e-7):
            raise RuntimeError('physical target reset differs from registered geometry')
