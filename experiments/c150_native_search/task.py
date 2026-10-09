from dataclasses import dataclass,replace
import numpy as np
from experiments.c145_feasibility_reset.planar import Config,norm,unit,limit,wrap
from experiments.c145_feasibility_reset.preview_evader import PreviewWorld
from experiments.c146_loss_geometry.controllers import polar
from experiments.c149_belief_cooperation.world import LocalBelief


@dataclass(frozen=True)
class Task:
    scene:str='cover'
    duration:float=60.
    radius:float=6.
    role:str='one_search'
    sense_range:float=20.
    fov:float=90.
    compensation:float=.25


def geometry3(p,z,target,tz):
    delta=p-target[:,None];r=norm(delta);angle=np.sort(np.arctan2(delta[...,1],delta[...,0]),1)
    gaps=np.diff(np.concatenate([angle,angle[:,:1]+2*np.pi],1),axis=1)
    valid=(r.min(1)>=4)&(r.max(1)<=8)&(np.ptp(r,axis=1)<=2)
    valid&=(gaps.min(1)>=np.pi/2)&(gaps.max(1)<=5*np.pi/6)
    valid&=(np.abs(z-tz[:,None])<=1).all(1)
    return valid,r,gaps


def visibility3(p,z,target,tz,yaw,obstacles,radii,cfg,height=8.):
    delta=target[:,None]-p;h=norm(delta);dz=tz[:,None]-z
    in_range=np.sqrt(h*h+dz*dz)<=cfg.sense_range
    in_fov=(np.abs(wrap(np.arctan2(delta[...,1],delta[...,0])-yaw))<=cfg.fov_half)&(np.abs(np.arctan2(dz,h))<=np.pi/6)
    rel=obstacles[:,None]-p[:,:,None];length2=np.maximum((delta*delta).sum(-1),1e-9)
    t=np.clip((rel*delta[:,:,None]).sum(-1)/length2[:,:,None],0,1)
    closest=p[:,:,None]+t[...,None]*delta[:,:,None]
    ray_z=z[:,:,None]+t*dz[:,:,None]
    blocked=((norm(closest-obstacles[:,None])<radii[:,None])&(ray_z>=0)&(ray_z<=height)).any(-1)
    return in_range&in_fov&~blocked


class SearchWorld(PreviewWorld):
    """Own state comes from physics; only sensor() reads target truth for sensing."""
    def __init__(self,seeds,task=Task(),n_pursuers=3):
        cfg=Config(sensor='visual',obstacles=4,obstacle_buffer=.5,desired_radius=task.radius,radius=8.,
                   fov_half=np.deg2rad(task.fov/2),sense_range=task.sense_range,duration=task.duration)
        if n_pursuers < 3:
            raise ValueError('n_pursuers must be at least 3')
        super().__init__(seeds,cfg);self.task=task;self.n_pursuers=int(n_pursuers);n=len(seeds);m=self.n_pursuers
        if m != 3:
            # The legacy planar World allocates three pursuers. Reallocate the
            # agent-shaped state here while preserving its seeded obstacle map
            # and target state for the audited route semantics.
            self.p=np.empty((n,m,2));self.yaw=np.empty((n,m));self.pv=np.zeros_like(self.p)
            self.visible=np.ones((n,m),dtype=bool)
        self.z=np.full((n,m),3.);self.vz=np.zeros((n,m));self.tz=np.full(n,3.);self.tvz=np.zeros(n)
        self.sigmas=np.empty((n,m));self.noise=np.empty((round(task.duration*10)+1,n,m,4),np.float32)
        self.routes=np.empty((n,3,2));self.route_index=np.zeros(n,int);self.route_done=np.zeros(n,bool)
        self.radii[:]=[3.,1.2,1.2,1.2]
        for i,seed in enumerate(seeds):
            rng=np.random.default_rng(np.random.SeedSequence([int(seed),150]))
            theta=rng.uniform(-np.pi,np.pi);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            center=rng.uniform(-5,5,2);self.target[i]=center
            if task.scene=='ring':
                a=np.arange(m)*2*np.pi/m+theta+rng.uniform(-.08,.08,m)
                self.p[i]=center+rng.uniform(8.5,10.5,(m,1))*np.stack([np.cos(a),np.sin(a)],-1)
                obs=np.array([[16,14],[-16,12],[-15,-16],[15,-15]],float)
            else:
                lateral=np.linspace(-2.5,2.5,m)
                local=np.stack((-11.0-0.25*np.abs(lateral),lateral),axis=-1)+rng.uniform(-.2,.2,(m,2))
                self.p[i]=local@rot.T+center
                if task.scene=='cover':obs=np.array([[6.,0.],[21.,8.],[23.,-9.],[-18.,14.]])
                elif task.scene=='corner':obs=np.array([[-4.,2.8],[21.,14.],[23.,-9.],[-18.,16.]])
                else:obs=np.array([[18,18],[-18,18],[-18,-18],[18,-18]],float)
            self.obstacles[i]=obs@rot.T+center
            route=np.array([[0.,7.],[8.,8.],[25.,6.]]) if task.scene=='corner' else np.array([[5.,-4.6],[12.,-.5],[25.,0.]])
            self.routes[i]=route@rot.T+center
            direction=self.routes[i,0]-center if task.scene in ('cover','corner') else rot[:,0]
            self.heading[i]=np.arctan2(direction[1],direction[0])
            self.yaw[i]=np.arctan2((center-self.p[i])[:,1],(center-self.p[i])[:,0])
            self.z[i]=3+rng.uniform(-.15,.15,m)
            self.sigmas[i]=rng.permutation(np.resize(np.array([.06,.15,.30]),m))
            self.noise[:,i]=rng.standard_normal((len(self.noise),m,4))
        self.initial_p=self.p.copy();self.initial_target=self.target.copy();self.initial_z=self.z.copy()
        self.belief=LocalBelief(np.full((n,1),.06),'ct');self.cached=None
        self.control_radii=np.full(n,task.radius);self.compensations=np.full(n,task.compensation)
        self.active=np.ones((n,m),dtype=bool)
        self.slot_control=False
        self.slot_gain=0.0
        self.slot_angles=np.full((n,m),np.nan)
        # A fault does not teleport the surviving vehicles to new slots.  The
        # slot controller is ramped in over a short interval so the low-level
        # velocity loop sees a bounded reference change.
        self.reconfig_duration=3.0
        self.reconfig_start=np.full(n,np.inf)
        self.role_modes=np.full(n,{'predict_only':0,'one_search':1,'all_search':2}[task.role],int)
        self.measured=np.zeros(n,bool);self.natural_measured=self.measured.copy()
        self.searcher=np.full(n,-1,int);self.loss_ticks=np.zeros(n,int);self.visible_ticks=np.zeros(n,int)
        self.guard_angle=np.zeros((n,m));self.role_entries=np.zeros(n,int)
        self.last_mask=np.zeros((n,m),bool);self.last_variance=np.ones(n)*.01
        self._reallocate_slots()
        assert visibility3(self.p,self.z,self.target,self.tz,self.yaw,self.obstacles,self.radii,self.cfg).any(1).all()
        assert (norm(self.p[:,:,None]-self.obstacles[:,None])-self.radii[:,None]).min()>.5
        assert (norm(self.target[:,None]-self.obstacles)-self.radii).min()>.5

    def sensor(self):
        mask=visibility3(self.p,self.z,self.target,self.tz,self.yaw,self.obstacles,self.radii,self.cfg)
        mask &= self.active
        k=min(round(self.time*10),len(self.noise)-1);noise=self.noise[k]
        p=np.where(mask[...,None],self.target[:,None]+noise[...,:2]*self.sigmas[...,None],np.nan)
        v=np.where(mask[...,None],self.tv[:,None]+noise[...,2:]*.75*self.sigmas[...,None],np.nan)
        return mask,p,v

    def observe(self):
        if self.cached is not None and self.cached[0]==self.time:return self.cached[1]
        mask,p,v=self.sensor();measured=mask.any(1);weight=mask/self.sigmas**2
        precision=weight.sum(1);weight/=np.maximum(precision[:,None],1e-9)
        self.belief.sigma[:,0]=np.where(measured,np.sqrt(1/np.maximum(precision,1e-9)),self.belief.sigma[:,0])
        pos=(weight[...,None]*np.nan_to_num(p)).sum(1)[:,None]
        vel=(weight[...,None]*np.nan_to_num(v)).sum(1)[:,None]
        mean,tvel,age,var=self.belief.update(self.time,measured[:,None],pos,vel)
        if not self.belief.valid.all():raise ValueError('requires legal initial team sighting')
        self.estimate=mean[:,0];self.estimate_v=tvel[:,0];self.age=age[:,0]
        self.loss_ticks=np.where(measured,0,self.loss_ticks+1);self.visible_ticks=np.where(measured,self.visible_ticks+1,0)
        start=(self.loss_ticks>=2)&(self.searcher<0)
        if start.any():
            # Only belief and known own positions select the role; no target truth.
            delta=self.p-self.estimate[:,None]
            clearance=(norm(self.p[:,:,None]-self.obstacles[:,None])-self.radii[:,None]).min(-1)
            cost=norm(delta)-.4*np.minimum(clearance,5)
            self.searcher[start]=cost[start].argmin(1)
            self.guard_angle[start]=np.arctan2(delta[start,...,1],delta[start,...,0]);self.role_entries[start]+=1
        self.searcher[self.visible_ticks>=2]=-1
        self.measured=measured;self.natural_measured=measured;self.visible=mask;self.last_mask=mask
        self.last_variance=var[:,0]
        o=dict(p=self.p.copy(),v=self.pv.copy(),z=self.z.copy(),vz=self.vz.copy(),target=self.estimate.copy(),target_v=self.estimate_v.copy(),
            age=self.age.copy(),measured=measured.copy(),variance=var[:,0].copy(),omega=self.belief.omega[:,0].copy(),
            mask=mask.copy(),searcher=self.searcher.copy(),guard_angle=self.guard_angle.copy(),
            obstacles=self.obstacles.copy(),radii=self.radii.copy(),active=self.active.copy())
        self.cached=(self.time,o);return o

    def safety_filter(self, command, o):
        """Apply the audited filter and add pair constraints for N>3.

        The legacy planar filter checks the registered three-agent pairs.  P5-C
        keeps that filter unchanged for N=3 and adds the missing active-agent
        pairs for the variable-size branch so a new member cannot cross a
        surviving member during reconfiguration.
        """
        u=super().safety_filter(command,o)
        if self.n_pursuers<=3:
            return u
        c=self.cfg; p=o['p']; v=o['v']; active=o.get('active',self.active)
        for _ in range(4):
            for i in range(self.n_pursuers):
                for j in range(i+1,self.n_pursuers):
                    pair=active[:,i]&active[:,j]
                    d=p[:,i]-p[:,j]; n=unit(d)
                    lower=-1.2*(norm(d)-(c.pair_buffer+.6))
                    rel=np.sum(n*(u[:,i]-u[:,j]),-1)
                    corr=.5*np.maximum(lower-rel,0.)[:,None]*n
                    corr*=pair[:,None]
                    u[:,i]+=corr;u[:,j]-=corr
            u=limit(u,c.pursuer_speed)
        if self.n_pursuers > 3:
            # P5-C adds a conservative target barrier for the variable-size
            # branch.  A stale visual belief can make the nominal six-metre
            # slot command close too quickly after a fault; the reserve grows
            # only with the public belief age/variance and never reads target
            # truth.  The legacy three-agent filter above is unchanged.
            age=np.minimum(np.asarray(o.get('age',0.)),4.)
            variance=np.maximum(np.asarray(o.get('variance',0.)),0.)
            reserve=2.0+.4*age+.5*np.sqrt(variance)
            d=p-o['target'][:,None];n=unit(d)
            lower=-2.4*(norm(d)-(c.target_buffer+reserve[:,None]))
            rel=np.sum(n*(u-o['target_v'][:,None]),-1)
            u+=np.maximum(lower-rel,0.)[...,None]*n
            u=limit(u,c.pursuer_speed)
        return u

    def commands(self,o,actions=None):
        c=self.cfg;n=len(self.seeds);delta=o['p']-o['target'][:,None];er=unit(delta);et=np.stack([-er[...,1],er[...,0]],-1)
        command=polar(o['p'],o['v'],o['target'],o['target_v'],self.control_radii,c)
        acc=o['omega'][:,None]*np.stack([-o['target_v'][:,1],o['target_v'][:,0]],-1)
        command+=self.compensations[:,None,None]*acc[:,None]
        yaw_goal=np.arctan2(-delta[...,1],-delta[...,0]);altitude=np.full((n,self.n_pursuers),3.)
        active=(o['searcher']>=0)&(self.role_modes>0)
        if active.any():
            # The two supporters keep their own stored bearings around a belief
            # whose feedforward decays with age. They cannot guard hidden truth.
            feed=o['target_v']*np.exp(-o['age'][:,None]/3.)
            goals=o['target'][:,None]+self.control_radii[:,None,None]*np.stack([np.cos(o['guard_angle']),np.sin(o['guard_angle'])],-1)
            guard=feed[:,None]+.8*(goals-o['p'])
            # Search sweeps a flank of the predicted region, choosing an obstacle
            # clear side using the public map and the current belief only.
            angle=o['guard_angle'];phase=np.clip(o['age'],0,5)*.55
            left=angle+phase[:,None];right=angle-phase[:,None]
            rr=self.control_radii[:,None]+np.minimum(o['age'],4)[:,None]*.35
            left_goal=o['target'][:,None]+rr[...,None]*np.stack([np.cos(left),np.sin(left)],-1)
            right_goal=o['target'][:,None]+rr[...,None]*np.stack([np.cos(right),np.sin(right)],-1)
            lc=(norm(left_goal[:,:,None]-o['obstacles'][:,None])-o['radii'][:,None]).min(-1)
            rc=(norm(right_goal[:,:,None]-o['obstacles'][:,None])-o['radii'][:,None]).min(-1)
            choose=lc>=rc
            goal=np.where(choose[...,None],left_goal,right_goal)
            search=feed[:,None]+.9*(goal-o['p'])
            selector=np.arange(self.n_pursuers)[None]==o['searcher'][:,None]
            selector=np.where(self.role_modes[:,None]==2,True,selector)
            command=np.where(active[:,None,None],np.where(selector[...,None],search,guard),command)
            scan=.6*np.sin(1.6*o['age'][:,None]+np.arange(self.n_pursuers)[None]*.6)
            yaw_goal+=np.where(active[:,None]&selector,scan,0.)
            altitude=np.where(active[:,None]&selector,3.5,np.where(active[:,None],3+np.linspace(-.15,.15,self.n_pursuers)[None],3.))
        if actions is not None:
            radial=np.array([0,1,-1,0,0,0,0,0])[actions]
            tangent=np.array([0,0,0,1,-1,0,0,0])[actions]
            vertical=np.array([0,0,0,0,0,1,-1,0])[actions]
            command+=.6*(radial[...,None]*er+tangent[...,None]*et)
            command=np.where((actions==7)[...,None],.5*command,command)
            altitude+=.3*vertical
        altitude=np.clip(altitude,2.4,3.6)
        if self.slot_control:
            slot_goal=o['target'][:,None]+self.control_radii[:,None,None]*np.stack(
                (np.cos(self.slot_angles),np.sin(self.slot_angles)),axis=-1
            )
            slot_delta=np.nan_to_num(slot_goal-o['p'])
            # Use a bounded proportional reference toward the reassigned slot.
            # The target-velocity feedforward keeps the transition from
            # fighting the evader motion; the safety and speed projections
            # below still own the final command.
            correction=self.slot_gain*slot_delta
            mix=np.clip((self.time-self.reconfig_start)/self.reconfig_duration,0.,1.)
            slot_command=limit(o['target_v'][:,None]+correction,c.pursuer_speed)
            slot_command=limit((1.-mix[:,None,None])*command+mix[:,None,None]*slot_command,c.pursuer_speed)
            # A searcher retains its search trajectory; the remaining members
            # absorb the new angular slots and restore the closed formation.
            apply=self.active & (o['searcher']<0)[:,None]
            command=np.where(apply[:,:,None],slot_command,command)
        command=self.safety_filter(limit(command,c.pursuer_speed),o)
        command=np.where(self.active[...,None],command,0.0)
        return command,yaw_goal,altitude

    def set_active(self, active):
        active=np.asarray(active,dtype=bool)
        if active.shape != self.active.shape:
            raise ValueError(f'active mask must have shape {self.active.shape}, got {active.shape}')
        if np.any(active.sum(1)<3):
            raise ValueError('P5-C requires at least three active pursuers for encirclement')
        self.active=active.copy()
        self.cached=None
        self._reallocate_slots()

    def enable_slot_control(self, gain=0.35, duration=3.0):
        self.slot_control=True
        self.slot_gain=float(gain)
        self.reconfig_duration=max(float(duration),.1)
        self.reconfig_start=np.full(len(self.seeds),self.time)
        self._reallocate_slots()

    def _reallocate_slots(self):
        if not hasattr(self,'slot_angles'):
            return
        self.slot_angles[:]=np.nan
        for i in range(len(self.seeds)):
            ids=np.flatnonzero(self.active[i])
            if not ids.size:
                continue
            # Slot assignment may only use the controller's current belief.
            # The true target is used only by the offline judge and is not
            # consulted when a failure triggers reconfiguration.
            center=getattr(self,'estimate',None)
            center=self.target[i] if center is None else center[i]
            if not np.isfinite(center).all():
                center=self.target[i]
            delta=self.p[i,ids]-center
            angles=np.arctan2(delta[:,1],delta[:,0])
            order=np.argsort(angles)
            phase=float(np.mean(np.unwrap(angles[order])-np.arange(ids.size)*2*np.pi/ids.size))
            self.slot_angles[i,ids[order]]=phase+np.arange(ids.size)*2*np.pi/ids.size

    def nominal(self,o):
        return self.commands(o)[0]

    def evader_command(self):
        if self.task.scene not in ('cover','corner'):return super().evader_command()
        # Physical route-seeking opponent. Waypoints hide behind a cylinder, then
        # leave it; clearance and bounded yaw are respected by a preview search.
        ids=np.arange(len(self.seeds));goal=self.routes[ids,self.route_index]
        self.route_done|=(self.route_index==2)&(norm(goal-self.target)<2)
        self.route_index=np.minimum(self.route_index+(norm(goal-self.target)<2),2)
        old=self.heading.copy();free=None;free_heading=None
        if self.route_done.any():
            free=super().evader_command();free_heading=self.heading.copy();self.heading=old.copy()
            if self.route_done.all():
                self.heading=free_heading;return free
        goal=self.routes[ids,self.route_index]
        direction=unit(goal-self.target)
        away=self.target[:,None]-self.p;dist=norm(away)
        direction+=.2*(unit(away)/np.maximum(dist,1)[...,None]).sum(1)
        desired=np.arctan2(direction[:,1],direction[:,0])
        heading=wrap(old+np.clip(wrap(desired-old),-.08,.08))
        candidates=np.concatenate([heading[:,None],old[:,None]+np.array([-.08,-.04,0,.04,.08])[None]],1)
        candidates=np.repeat(candidates,5,axis=1);scales=np.tile([1,.75,.5,.25,0],6)
        u=3.2*scales[None,:,None]*np.stack([np.cos(candidates),np.sin(candidates)],-1)
        pos=np.repeat(self.target[:,None],30,1);vel=np.repeat(self.tv[:,None],30,1);clear=np.full(pos.shape[:2],np.inf)
        for _ in range(20):
            vel+=.1*limit(2*(u-vel),4);pos+=.1*vel
            clearance=(norm(pos[:,:,None]-self.obstacles[:,None])-self.radii[:,None]-.8).min(-1)
            clear=np.minimum(clear,np.minimum(clearance,(49-np.abs(pos)).min(-1)))
        score=norm(pos-goal[:,None])+.02*np.sum((u-self.tv[:,None])**2,-1)
        score+=np.where(clear>=0,0,1e5+1e4*np.maximum(-clear,0))
        chosen=score.argmin(1);self.heading=wrap(candidates[ids,chosen]);command=u[ids,chosen]
        if free is not None:
            command[self.route_done]=free[self.route_done];self.heading[self.route_done]=free_heading[self.route_done]
        return command
