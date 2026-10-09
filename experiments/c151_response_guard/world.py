import numpy as np
from experiments.c150_native_search.task import SearchWorld,Task,visibility3
from experiments.c145_feasibility_reset.planar import norm,wrap
from .guard import response_guard

METHODS={'old_predict':(0,False,False),'old_search':(1,False,False),
         'brake_predict':(0,True,False),'brake_search':(1,True,False),'staged_brake':(1,True,True)}

class ResponseWorld(SearchWorld):
    def __init__(self,seeds,task=Task(),scale=1.,n_pursuers=3):
        super().__init__(seeds,task,n_pursuers=n_pursuers);n=len(seeds);m=self.n_pursuers;self.scale=scale
        self.p=self.target[:,None]+scale*(self.p-self.target[:,None])
        self.obstacles=self.target[:,None]+scale*(self.obstacles-self.target[:,None])
        self.routes=self.target[:,None]+scale*(self.routes-self.target[:,None]);self.radii*=scale
        self.initial_p=self.p.copy()
        if not visibility3(self.p,self.z,self.target,self.tz,self.yaw,self.obstacles,self.radii,self.cfg).any(1).all():
            raise ValueError('scaled scene lacks an initial legal team observation')
        self.braking=np.zeros(n,bool);self.staged=np.zeros(n,bool)
        self.stage_role=np.full(n,-1,int);self.stage_angles=np.zeros((n,m));self.stage_on=np.zeros(n,bool)
        self.stage_mix=np.zeros(n);self.stage_clock=None;self.stage_entries=np.zeros(n,int)
        self.raw_command=np.zeros((n,m,2));self.guard_delta=np.zeros((n,m,2))

    def configure(self,sl,name):
        role,brake,stage=METHODS[name];self.role_modes[sl]=role;self.braking[sl]=brake;self.staged[sl]=stage

    def observe(self):
        if self.stage_clock==self.time and self.cached is not None:return self.cached[1]
        o=super().observe();start=self.staged&~self.stage_on&~o['measured']&(o['age']>=1.5-1e-8)
        if start.any():
            d=o['p']-o['target'][:,None]
            clear=(norm(o['p'][:,:,None]-o['obstacles'][:,None])-o['radii'][:,None]).min(-1)
            cost=norm(d)-.4*np.minimum(clear,5)
            self.stage_role[start]=cost[start].argmin(1);self.stage_angles[start]=np.arctan2(d[start,...,1],d[start,...,0])
            self.stage_on[start]=True;self.stage_entries[start]+=1
        self.stage_on[self.staged&(self.visible_ticks>=2)]=False
        dt=0. if self.stage_clock is None else max(0.,self.time-self.stage_clock)
        goal=self.stage_on.astype(float)
        self.stage_mix+=np.clip(goal-self.stage_mix,-dt/.5,dt/.5)
        self.stage_role[self.staged&~self.stage_on&(self.stage_mix<=1e-8)]=-1
        o=dict(o);o['searcher']=o['searcher'].copy();o['guard_angle']=o['guard_angle'].copy()
        o['searcher'][self.staged]=self.stage_role[self.staged];o['guard_angle'][self.staged]=self.stage_angles[self.staged]
        o['phase_mix']=np.where(self.staged,self.stage_mix,(o['searcher']>=0)&(self.role_modes>0))
        o['deployed_searcher']=np.where(self.role_modes>0,o['searcher'],-1)
        self.stage_clock=self.time;self.cached=(self.time,o);return o

    def commands(self,o,actions=None):
        # Stage state advances only in observe(), not in feature/control queries.
        u,y,z=super().commands(o,actions)
        if self.staged.any():
            base=dict(o,searcher=np.full(len(self.seeds),-1,int))
            bu,by,bz=super().commands(base,actions);mix=self.stage_mix
            blended=bu+mix[:,None,None]*(u-bu)
            u=np.where(self.staged[:,None,None],blended,u)
            y=np.where(self.staged[:,None],by+mix[:,None]*wrap(y-by),y)
            z=np.where(self.staged[:,None],bz+mix[:,None]*(z-bz),z)
        self.raw_command=u.copy()
        if self.braking.any():
            # Variable-size teams use a larger belief-space reserve because a
            # slot handoff can leave an active member with braking distance
            # while the low-level Isaac controller catches up.  The reserve is
            # applied to the shared estimate only; it never reads target truth.
            target_buffer=3.0 if self.n_pursuers>3 else 1.5
            new=response_guard(u,o,self.cfg.pursuer_speed,target_buffer=target_buffer)
            u=np.where(self.braking[:,None,None],new,u)
        # A failed member must remain physically inactive even when the
        # response guard performs pair/obstacle corrections on the survivors.
        u=np.where(self.active[...,None],u,0.0)
        self.guard_delta=u-self.raw_command
        return u,y,z
