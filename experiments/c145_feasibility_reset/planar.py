"""Independent, CPU-vectorized encirclement feasibility model (not Isaac).

All units are SI. Acceleration-limited velocity tracking approximates the
existing 2/s planar response, but does not claim rigid-body equivalence.
Controllers receive only measurement/belief objects; truth is used by the
evader, dynamics and scorer. No legacy checkpoints or results are modified.
"""
from dataclasses import dataclass, asdict
import numpy as np


def norm(x):
    return np.linalg.norm(x, axis=-1)


def unit(x):
    return x / np.maximum(norm(x)[..., None], 1e-9)


def limit(x, maximum):
    return x * np.minimum(1., np.asarray(maximum) / np.maximum(norm(x), 1e-9))[..., None]


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Config:
    dt: float = .1
    substeps: int = 5
    duration: float = 30.
    pursuer_speed: float = 4.
    evader_ratio: float = .8
    acceleration: float = 4.
    velocity_kp: float = 2.
    yaw_rate: float = 1.5
    target_turn_rate: float = .8
    arena_half: float = 50.
    radius: float = 5.5
    desired_radius: float = 4.2
    target_buffer: float = 1.5
    pair_buffer: float = 1.
    obstacle_buffer: float = .4
    max_gap: float = np.pi * 150 / 180
    sense_range: float = 100.
    fov_half: float = np.pi / 3
    obstacles: int = 0
    layout: str = 'ring'
    evader: str = 'smooth_gap'
    sensor: str = 'shared'
    predictor: str = 'cv'
    blackout_start: float = 10.
    blackout_duration: float = 0.
    radial_gain: float = .9
    angular_gain: float = .65
    controller: str = 'polar'
    safety: bool = True


def geometry(p, target, cfg):
    d = norm(p - target[:, None])
    angles = np.sort(np.arctan2((p-target[:, None])[..., 1],
                               (p-target[:, None])[..., 0]), axis=1)
    gaps = np.diff(np.concatenate([angles, angles[:, :1]+2*np.pi], 1), axis=1)
    return ((d.max(1) <= cfg.radius) & (d.min(1) >= cfg.target_buffer)
            & (gaps.max(1) <= cfg.max_gap)), d, gaps.max(1)


def visibility(p, target, yaw, obstacles, radii, cfg):
    delta = target[:, None] - p
    distance = norm(delta)
    in_range = distance <= cfg.sense_range
    in_fov = np.abs(wrap(np.arctan2(delta[..., 1], delta[..., 0])-yaw)) <= cfg.fov_half
    blocked = np.zeros(distance.shape, bool)
    if obstacles.shape[1]:
        rel = obstacles[:, None] - p[:, :, None]
        length2 = np.maximum(np.sum(delta*delta, -1), 1e-9)
        t = np.clip(np.sum(rel*delta[:, :, None], -1)/length2[:, :, None], 0, 1)
        closest = p[:, :, None]+t[..., None]*delta[:, :, None]
        blocked = (norm(closest-obstacles[:, None]) < radii[:, None]).any(2)
    return in_range & in_fov & ~blocked, in_range, in_fov, blocked


class World:
    def __init__(self, seeds, cfg=Config()):
        self.cfg, self.seeds = cfg, np.asarray(seeds, dtype=np.int64)
        if cfg.layout not in ('ring', 'arc') or cfg.evader not in ('cv','turn','smooth_gap','hard_gap'):
            raise ValueError('unregistered scenario')
        if cfg.sensor not in ('shared','visual') or cfg.predictor not in ('cv','last'):
            raise ValueError('unregistered information mode')
        b = len(seeds)
        self.p = np.empty((b, 3, 2)); self.target = np.empty((b, 2))
        self.yaw = np.empty((b, 3)); self.heading = np.empty(b)
        self.obstacles = np.empty((b, cfg.obstacles, 2)); self.radii = np.empty((b, cfg.obstacles))
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            t = rng.uniform(-10, 10, 2); phase = rng.uniform(-np.pi,np.pi)
            if cfg.layout == 'ring':
                angles = phase + np.arange(3)*2*np.pi/3 + rng.uniform(-.22,.22,3)
            else:
                angles = phase + np.array([-.65,0,.65]) + rng.uniform(-.08,.08,3)
            self.p[i] = t + rng.uniform(8,12,(3,1))*np.stack([np.cos(angles),np.sin(angles)],-1)
            self.target[i] = t; self.heading[i] = rng.uniform(-np.pi,np.pi)
            self.yaw[i] = np.arctan2(t[1]-self.p[i,:,1],t[0]-self.p[i,:,0])
            for j in range(cfg.obstacles):
                for _ in range(10000):
                    pos = rng.uniform(-35,35,2); radius = rng.uniform(.7,1.7)
                    if (norm(self.p[i]-pos).min()>radius+2 and norm(t-pos)>radius+4
                        and (j==0 or np.all(norm(self.obstacles[i,:j]-pos)>self.radii[i,:j]+radius+2))):
                        self.obstacles[i,j]=pos;self.radii[i,j]=radius;break
                else: raise RuntimeError('obstacle rejection sampler exhausted')
        self.pv=np.zeros_like(self.p);self.tv=np.zeros_like(self.target)
        self.estimate=self.target.copy();self.estimate_v=self.tv.copy()
        self.last=self.target.copy();self.last_v=self.tv.copy();self.age=np.zeros(b)
        self.time=0.;self.visible=np.ones((b,3),bool);self.measured=np.ones(b,bool)
        self.unsafe=np.zeros(b,bool);self.min_target=np.full(b,np.inf)
        self.min_pair=np.full(b,np.inf);self.min_obstacle=np.full(b,np.inf)
        self.run_geometry=np.zeros(b,int);self.run_visual=np.zeros(b,int)
        self.max_geometry=np.zeros(b,int);self.max_visual=np.zeros(b,int)
        self.hit_geometry=np.full(b,np.nan);self.hit_visual=np.full(b,np.nan)
        self.hold2_time=np.full(b,np.nan);self.hold5_time=np.full(b,np.nan)
        self.team_lost_steps=np.zeros(b,int);self.individual_lost_steps=np.zeros(b,int)
        self.collision_target=np.zeros(b,bool);self.collision_agent=np.zeros(b,bool)
        self.collision_obstacle=np.zeros(b,bool);self.out_of_bounds=np.zeros(b,bool)
        self.target_obstacle_invalid=np.zeros(b,bool)
        self.geometry_steps=np.zeros(b,int);self.late_geometry_steps=np.zeros(b,int)
        self.blackout_geometry_steps=np.zeros(b,int);self.blackout_steps=0
        self.post_blackout_recovery=np.full(b,np.nan)
        self.gap_phase=np.zeros(b);self.gap_initialized=np.zeros(b,bool)
        self.initial_p=self.p.copy();self.initial_target=self.target.copy()
        # Fixed slots keep the initial cyclic assignment and a world-fixed phase.
        th=np.arctan2((self.p-self.target[:,None])[...,1],(self.p-self.target[:,None])[...,0])
        unwrapped=np.unwrap(th,axis=1)
        self.slot_phase=np.mean(unwrapped-np.arange(3)*2*np.pi/3,axis=1)

    def observe(self):
        cfg=self.cfg
        self.visible,*_=visibility(self.p,self.target,self.yaw,self.obstacles,self.radii,cfg)
        self.measured=self.visible.any(1) if cfg.sensor=='visual' else np.ones(len(self.seeds),bool)
        blackout=cfg.blackout_start <= self.time < cfg.blackout_start+cfg.blackout_duration-1e-7
        if blackout:self.measured[:]=False
        self.last[self.measured]=self.target[self.measured]
        self.last_v[self.measured]=self.tv[self.measured]
        self.age=np.where(self.measured,0.,self.age+cfg.dt)
        self.estimate=self.last.copy()
        if cfg.predictor=='cv':self.estimate+=self.age[:,None]*self.last_v
        self.estimate_v=self.last_v.copy() if cfg.predictor=='cv' else np.where(self.measured[:,None],self.last_v,0.)
        return dict(p=self.p.copy(),v=self.pv.copy(),target=self.estimate.copy(),
                    target_v=self.estimate_v.copy(),age=self.age.copy(),
                    measured=self.measured.copy(),obstacles=self.obstacles.copy(),radii=self.radii.copy())

    def evader_command(self):
        c=self.cfg;speed=c.pursuer_speed*c.evader_ratio
        if c.evader=='turn': desired_heading=self.heading+.30*c.dt
        elif c.evader=='cv':desired_heading=self.heading.copy()
        else:
            angle=np.sort(np.arctan2((self.p-self.target[:,None])[...,1],(self.p-self.target[:,None])[...,0]),axis=1)
            gaps=np.diff(np.concatenate([angle,angle[:,:1]+2*np.pi],axis=1),axis=1)
            k=np.argmax(gaps,1);chosen=angle[np.arange(len(k)),k]+gaps[np.arange(len(k)),k]/2
            # Gap direction has discontinuities near ties. The smooth opponent
            # has an explicit desired-heading slew limit; both share accel caps.
            intent=.8*np.stack([np.cos(chosen),np.sin(chosen)],-1)+.2*np.stack([np.cos(self.heading),np.sin(self.heading)],-1)
            desired_heading=np.arctan2(intent[:,1],intent[:,0])
        direction=np.stack([np.cos(desired_heading),np.sin(desired_heading)],-1)
        if c.evader.endswith('gap'):
            away=self.target[:,None]-self.p;distance=norm(away)
            direction+=(unit(away)/np.maximum(distance-1.,.5)[...,None]*(distance<30)[...,None]).sum(1)
        if c.obstacles:
            away=self.target[:,None]-self.obstacles; clearance=norm(away)-self.radii
            direction+=(unit(away)*(.8/np.maximum(clearance,.1)*(clearance<8))[...,None]).sum(1)
        inner=c.arena_half-8.
        direction+=np.where(self.target < -inner,(-inner-self.target)/8,0)
        direction-=np.where(self.target > inner,(self.target-inner)/8,0)
        direction=np.where(self.target < -inner,np.maximum(direction,.25),direction)
        direction=np.where(self.target > inner,np.minimum(direction,-.25),direction)
        desired_heading=np.arctan2(direction[:,1],direction[:,0])
        if c.evader=='smooth_gap':
            self.heading=wrap(self.heading+np.clip(wrap(desired_heading-self.heading),-c.target_turn_rate*c.dt,c.target_turn_rate*c.dt))
        else:self.heading=desired_heading
        return speed*np.stack([np.cos(self.heading),np.sin(self.heading)],-1)

    def nominal(self, obs):
        c=self.cfg;delta=obs['p']-obs['target'][:,None];r=norm(delta)
        if c.controller=='chase':command=-.8*delta+obs['target_v'][:,None]
        elif c.controller=='slots':
            a=self.slot_phase[:,None]+np.arange(3)*2*np.pi/3
            goal=obs['target'][:,None]+c.desired_radius*np.stack([np.cos(a),np.sin(a)],-1)
            command=obs['target_v'][:,None]+c.radial_gain*(goal-obs['p'])
        elif c.controller=='polar':
            er=unit(delta);et=np.stack([-er[...,1],er[...,0]],-1)
            a=np.arctan2(delta[...,1],delta[...,0]);order=np.argsort(a,axis=1)
            sa=np.take_along_axis(a,order,axis=1)
            forward=(np.roll(sa,-1,1)-sa)%(2*np.pi)
            back=np.roll(forward,1,1)
            tangential=np.take_along_axis(c.angular_gain*(forward-back),np.argsort(order,axis=1),axis=1)*r
            command=obs['target_v'][:,None]-c.radial_gain*(r-c.desired_radius)[...,None]*er+tangential[...,None]*et
        else:raise ValueError('unregistered controller')
        return limit(command,c.pursuer_speed)

    def safety_filter(self, command, obs):
        """Alternating half-space corrections to velocity commands.

        An approximate planar barrier filter, not a QP or safety certificate.
        The actual acceleration-limited path is scored at every substep.
        All target constraints use the available estimate, including blackout.
        """
        c=self.cfg
        if not c.safety:return limit(command,c.pursuer_speed)
        u=command.copy();p=obs['p'];gain=1.2
        for _ in range(5):
            d=p-obs['target'][:,None];n=unit(d)
            lower=-gain*(norm(d)-(c.target_buffer+.65))
            rel=np.sum(n*(u-obs['target_v'][:,None]),-1)
            u+=np.maximum(lower-rel,0)[...,None]*n
            for i,j in ((0,1),(0,2),(1,2)):
                d=p[:,i]-p[:,j];n=unit(d)
                lower=-gain*(norm(d)-(c.pair_buffer+.6))
                rel=np.sum(n*(u[:,i]-u[:,j]),-1)
                correction=.5*np.maximum(lower-rel,0)[:,None]*n
                u[:,i]+=correction;u[:,j]-=correction
            # The registered C179/C180 density profiles can contain 4, 6, or
            # 8 obstacles.  Use the actual observation cardinality here so
            # every physical/analytic obstacle participates in the early
            # radial projection; the legacy four-obstacle path is unchanged.
            for j in range(obs['obstacles'].shape[1]):
                d=p-obs['obstacles'][:,None,j];n=unit(d)
                lower=-gain*(norm(d)-(obs['radii'][:,None,j]+c.obstacle_buffer+.8))
                u+=np.maximum(lower-np.sum(n*u,-1),0)[...,None]*n
            u=np.minimum(u,gain*(c.arena_half-1-p))
            u=np.maximum(u,-gain*(c.arena_half-1+p))
            u=limit(u,c.pursuer_speed)
        return u

    def step(self, obs, residual=None):
        c=self.cfg;command=self.nominal(obs)
        if residual is not None:command=limit(command+np.asarray(residual),c.pursuer_speed)
        command=self.safety_filter(command,obs);target_command=self.evader_command()
        dt=c.dt/c.substeps
        for _ in range(c.substeps):
            self.pv=limit(self.pv+dt*limit(c.velocity_kp*(command-self.pv),c.acceleration),c.pursuer_speed)
            self.tv=limit(self.tv+dt*limit(c.velocity_kp*(target_command-self.tv),c.acceleration),c.pursuer_speed*c.evader_ratio)
            self.p+=self.pv*dt;self.target+=self.tv*dt
            d=norm(self.p-self.target[:,None]).min(1);self.min_target=np.minimum(self.min_target,d)
            pair=np.stack([norm(self.p[:,i]-self.p[:,j]) for i,j in ((0,1),(0,2),(1,2))],1).min(1)
            self.min_pair=np.minimum(self.min_pair,pair)
            self.collision_target|=d<c.target_buffer;self.collision_agent|=pair<c.pair_buffer
            if c.obstacles:
                clearance=(norm(self.p[:,:,None]-self.obstacles[:,None])-self.radii[:,None]).min((1,2))
                self.min_obstacle=np.minimum(self.min_obstacle,clearance)
                self.collision_obstacle|=clearance<c.obstacle_buffer
                self.target_obstacle_invalid|=(norm(self.target[:,None]-self.obstacles)-self.radii).min(1)<.2
            self.out_of_bounds|=(np.abs(self.p)>c.arena_half).any((1,2))|(np.abs(self.target)>c.arena_half).any(1)
        goal_yaw=np.arctan2((obs['target'][:,None]-self.p)[...,1],(obs['target'][:,None]-self.p)[...,0])
        self.yaw=wrap(self.yaw+np.clip(wrap(goal_yaw-self.yaw)*2.,-c.yaw_rate,c.yaw_rate)*c.dt)
        self.time=round(self.time+c.dt,8)
        self.unsafe=self.collision_target|self.collision_agent|self.collision_obstacle|self.out_of_bounds|self.target_obstacle_invalid
        geom,dist,gap=geometry(self.p,self.target,c)
        vis,*_=visibility(self.p,self.target,self.yaw,self.obstacles,self.radii,c)
        visual=geom&vis.all(1)
        self.geometry_steps+=(geom&~self.unsafe)
        if self.time>c.duration-10.+1e-7:self.late_geometry_steps+=(geom&~self.unsafe)
        if c.blackout_start<self.time<=c.blackout_start+c.blackout_duration+1e-7:
            self.blackout_geometry_steps+=(geom&~self.unsafe);self.blackout_steps+=1
        if c.blackout_duration>0 and self.time>=c.blackout_start+c.blackout_duration-1e-7:
            idx=geom&~self.unsafe&np.isnan(self.post_blackout_recovery)
            self.post_blackout_recovery[idx]=max(0.,self.time-c.blackout_start-c.blackout_duration)
        self.run_geometry=np.where(geom&~self.unsafe,self.run_geometry+1,0)
        self.run_visual=np.where(visual&~self.unsafe,self.run_visual+1,0)
        self.max_geometry=np.maximum(self.max_geometry,self.run_geometry)
        self.max_visual=np.maximum(self.max_visual,self.run_visual)
        for mask, dest in ((geom&~self.unsafe,self.hit_geometry),(visual&~self.unsafe,self.hit_visual),
                           (self.run_geometry*c.dt>=2.-1e-8,self.hold2_time),(self.run_geometry*c.dt>=5.-1e-8,self.hold5_time)):
            idx=mask&np.isnan(dest);dest[idx]=self.time
        self.team_lost_steps+=(~self.measured);self.individual_lost_steps+=(~vis).sum(1)
        return dict(geometry=geom,visual=visual,unsafe=self.unsafe.copy(),dist=dist,gap=gap,command=command.copy())

    def records(self):
        records=[]
        for i,s in enumerate(self.seeds):
            def nullable(x):return float(x) if np.isfinite(x) else None
            records.append(dict(seed=int(s),first_geometry_s=nullable(self.hit_geometry[i]),first_visual_s=nullable(self.hit_visual[i]),
                hold2_time_s=nullable(self.hold2_time[i]),hold5_time_s=nullable(self.hold5_time[i]),
                max_geometry_hold_s=float(self.max_geometry[i]*self.cfg.dt),max_visual_hold_s=float(self.max_visual[i]*self.cfg.dt),
                safe_full_episode=bool(~self.unsafe[i]),target_buffer_violation=bool(self.collision_target[i]),
                agent_buffer_violation=bool(self.collision_agent[i]),obstacle_buffer_violation=bool(self.collision_obstacle[i]),
                target_obstacle_invalid=bool(self.target_obstacle_invalid[i]),boundary_violation=bool(self.out_of_bounds[i]),
                min_target_distance_m=nullable(self.min_target[i]),min_agent_distance_m=nullable(self.min_pair[i]),
                min_obstacle_clearance_m=nullable(self.min_obstacle[i]),team_loss_s=float(self.team_lost_steps[i]*self.cfg.dt),
                geometry_fraction=float(self.geometry_steps[i]/round(self.cfg.duration/self.cfg.dt)),
                late_geometry_fraction=float(self.late_geometry_steps[i]/round(min(10,self.cfg.duration)/self.cfg.dt)),
                blackout_geometry_fraction=float(self.blackout_geometry_steps[i]/self.blackout_steps) if self.blackout_steps else None,
                recovery_after_blackout_s=nullable(self.post_blackout_recovery[i]),
                individual_loss_fraction=float(self.individual_lost_steps[i]/(3*round(self.cfg.duration/self.cfg.dt)))))
        return records


def run(seeds, cfg, trace_count=0):
    w=World(seeds,cfg);trace=[]
    for _ in range(round(cfg.duration/cfg.dt)):
        obs=w.observe();info=w.step(obs)
        if trace_count:
            trace.append(np.concatenate([w.p[:trace_count].reshape(trace_count,6),w.target[:trace_count],
                                         info['geometry'][:trace_count,None],info['visual'][:trace_count,None],
                                         info['unsafe'][:trace_count,None]],axis=1))
    records=w.records();safe=np.array([r['safe_full_episode'] for r in records])
    summary=dict(n=len(seeds),first_geometry=float(np.isfinite(w.hit_geometry).mean()),first_visual=float(np.isfinite(w.hit_visual).mean()),
        safe_hold2=float(((w.max_geometry*cfg.dt>=2-1e-8)&safe).mean()),
        safe_hold5=float(((w.max_geometry*cfg.dt>=5-1e-8)&safe).mean()),
        safe_visual_hold5=float(((w.max_visual*cfg.dt>=5-1e-8)&safe).mean()),
        unsafe=float((~safe).mean()),mean_team_loss_s=float(np.mean(w.team_lost_steps*cfg.dt)),
        mean_late_geometry_fraction=float(np.mean(w.late_geometry_steps)/round(min(10,cfg.duration)/cfg.dt)))
    return dict(config=asdict(cfg),summary=summary,episodes=records),np.array(trace)
