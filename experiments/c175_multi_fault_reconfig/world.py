"""Active-only control and threat semantics; inherited sensor/route models.

A fault means task withdrawal with a functioning hover controller, not motor
failure or a crash. The withdrawn physical body remains a separation hazard.
"""
import copy

import numpy as np

from experiments.c145_feasibility_reset.planar import norm, unit, limit, wrap
from experiments.c150_native_search.task import Task
from experiments.c146_loss_geometry.controllers import polar
from experiments.c151_response_guard.guard import response_guard
from experiments.c171_variable_scale.world import VariableCornerWorld, VariableCorridorWorld

METHODS = ('active_polar', 'retained_slots', 'dynamic', 'greedy', 'no_feedforward', 'no_response_guard', 'gated_dynamic', 'hysteretic_dynamic', 'predictive_hysteretic')


class AuditMixin:
    method = 'dynamic'

    def _estimate_geometry_invalid(self, o):
        """Check formation stability using only the legal shared estimate."""
        p = np.asarray(o['p'])
        target = np.asarray(o['target'])
        active = np.asarray(o.get('active', self.active), dtype=bool)
        invalid = np.zeros(len(p), dtype=bool)
        for i in range(len(p)):
            ids = np.flatnonzero(active[i])
            if ids.size < 3:
                invalid[i] = True
                continue
            d = p[i, ids] - target[i]
            r = np.linalg.norm(d, axis=-1)
            a = np.sort(np.arctan2(d[:, 1], d[:, 0]))
            gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
            nominal = 2 * np.pi / ids.size
            invalid[i] = not (
                r.min() >= 4.0 and r.max() <= 8.0 and np.ptp(r) <= 2.0
                and gaps.min() >= .75 * nominal and gaps.max() <= 1.25 * nominal
            )
        return invalid

    def _update_reconfig_gate(self, o):
        if self.method not in ('gated_dynamic', 'hysteretic_dynamic', 'predictive_hysteretic') or not getattr(self, 'slot_control', False):
            return
        invalid = self._estimate_geometry_invalid(o)
        if self.method == 'hysteretic_dynamic':
            self.gate_invalid_ticks = np.where(invalid, self.gate_invalid_ticks + 1, 0)
            self.gate_valid_ticks = np.where(~invalid, self.gate_valid_ticks + 1, 0)
            newly = (~self.gate_active) & (self.gate_invalid_ticks >= 10)
            recovered = self.gate_active & (self.gate_valid_ticks >= 10)
            if newly.any():
                self.gate_active[newly] = True
                self.reconfig_start[newly] = self.time
            if recovered.any():
                self.gate_active[recovered] = False
                self.reconfig_start[recovered] = np.inf
        else:
            self.gate_invalid_ticks = np.where(invalid, self.gate_invalid_ticks + 1, 0)
            newly = (~self.gate_active) & (self.gate_invalid_ticks >= 10)
            if newly.any():
                self.gate_active[newly] = True
                self.reconfig_start[newly] = self.time

    def observe(self):
        o = super().observe()
        # Parent role selectors predate dropout. Repair their choice using only
        # legal belief, map and surviving own positions, including cached roles.
        cost = norm(o['p'] - o['target'][:, None])
        clear = (norm(o['p'][:, :, None] - o['obstacles'][:, None]) - o['radii'][:, None]).min(-1)
        choice = np.where(self.active, cost - .4 * np.minimum(clear, 5), np.inf).argmin(1)
        for name in ('searcher', 'stage_role'):
            role = getattr(self, name)
            bad = (role >= 0) & ~self.active[np.arange(len(role)), np.maximum(role, 0)]
            role[bad] = choice[bad]
        role = o['searcher'].copy()
        bad = (role >= 0) & ~self.active[np.arange(len(role)), np.maximum(role, 0)]
        role[bad] = choice[bad]
        o = dict(o, searcher=role, deployed_searcher=np.where(self.role_modes > 0, role, -1))
        self.cached = (self.time, o)
        return o

    def _reallocate_slots(self):
        if not hasattr(self, 'slot_angles'):
            return
        self.slot_angles[:] = np.nan
        if not hasattr(self, 'belief') or not self.belief.valid.all():
            return  # Initial angles are allocated after the first legal sighting.
        if not np.isfinite(self.estimate).all():
            raise ValueError('slot allocation requires a finite legal belief')
        for i, mask in enumerate(self.active):
            ids = np.flatnonzero(mask)
            d = self.p[i, ids] - self.estimate[i]
            angles = np.arctan2(d[:, 1], d[:, 0])
            order = np.argsort(angles)
            uniform = np.arange(len(ids)) * 2 * np.pi / len(ids)
            phase = np.mean(np.unwrap(angles[order]) - uniform)
            self.slot_angles[i, ids[order]] = phase + uniform

    def _compact(self, o):
        if not np.all(self.active == self.active[:1]):
            raise ValueError('this paired protocol uses one common fault index per batch')
        ids = np.flatnonzero(self.active[0])
        view = copy.copy(self)
        view.n_pursuers = len(ids)
        for key in ('p', 'pv', 'z', 'vz', 'yaw', 'active', 'slot_angles'):
            setattr(view, key, getattr(self, key)[:, ids].copy())
        view.departed_p = self.p[:, ~self.active[0]].copy()
        view.departed_v = self.pv[:, ~self.active[0]].copy()
        obs = dict(o)
        for key in ('p', 'v', 'z', 'vz', 'mask', 'active', 'guard_angle'):
            obs[key] = o[key][:, ids].copy()
        inverse = np.full(self.n_pursuers, -1, int)
        inverse[ids] = np.arange(len(ids))
        obs['searcher'] = np.where(o['searcher'] >= 0, inverse[np.maximum(o['searcher'], 0)], -1)
        return view, obs, ids

    def safety_filter(self, command, o):
        """Same all-constraint filter for all methods, including active-polar.

        Unlike C171, pair corrections are inside the obstacle/target projection
        loop, and there are no fictitious formation constraints on retired IDs.
        """
        c = self.cfg
        u = command.copy()
        p = o['p']
        for _ in range(8):
            d = p - o['target'][:, None]
            normal = unit(d)
            reserve = 2. + .4*np.minimum(o['age'], 4.) + .5*np.sqrt(np.maximum(o['variance'], 0.))
            lower = -2.4*(norm(d) - (c.target_buffer + reserve[:, None]))
            u += np.maximum(lower - (normal*(u-o['target_v'][:, None])).sum(-1), 0)[..., None]*normal
            for i in range(p.shape[1]):
                for j in range(i+1, p.shape[1]):
                    d = p[:, i] - p[:, j]
                    normal = unit(d)
                    lower = -1.2*(norm(d)-(c.pair_buffer+.6))
                    correction = .5*np.maximum(lower-(normal*(u[:, i]-u[:, j])).sum(-1), 0)[:, None]*normal
                    u[:, i] += correction
                    u[:, j] -= correction
            for j in range(o['obstacles'].shape[1]):
                d = p-o['obstacles'][:, None, j]
                normal = unit(d)
                lower = -1.2*(norm(d)-(o['radii'][:, None, j]+c.obstacle_buffer+.8))
                u += np.maximum(lower-(normal*u).sum(-1), 0)[..., None]*normal
            for j in range(getattr(self, 'departed_p', np.empty((0, 0))).shape[1]):
                d = p-self.departed_p[:, None, j]
                normal = unit(d)
                lower = -1.2*(norm(d)-1.6)
                relative = (normal*(u-self.departed_v[:, None, j])).sum(-1)
                u += np.maximum(lower-relative, 0)[..., None]*normal
            u = np.minimum(u, 1.2*(c.arena_half-1-p))
            u = np.maximum(u, -1.2*(c.arena_half-1+p))
            u = limit(u, c.pursuer_speed)
        return u

    def _core_commands(self,o,actions=None):
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
            # Legal CBC-TP-style short-horizon extrapolation. It uses only
            # the shared position/velocity estimate and loss age.
            slot_center = o['target']
            if self.method == 'predictive_hysteretic':
                horizon = np.clip(0.5 * o['age'], 0.0, 1.0)
                slot_center = slot_center + horizon[:, None] * o['target_v']
            slot_goal=slot_center[:,None]+self.control_radii[:,None,None]*np.stack(
                (np.cos(self.slot_angles),np.sin(self.slot_angles)),axis=-1
            )
            slot_delta=np.nan_to_num(slot_goal-o['p'])
            # Use a bounded proportional reference toward the reassigned slot.
            # The target-velocity feedforward keeps the transition from
            # fighting the evader motion; the safety and speed projections
            # below still own the final command.
            correction=self.slot_gain*slot_delta
            mix=np.clip((self.time-self.reconfig_start)/self.reconfig_duration,0.,1.)
            feedforward = 0. if self.method == 'no_feedforward' else 1.
            slot_command=limit(feedforward*o['target_v'][:,None]+correction,c.pursuer_speed)
            slot_command=limit((1.-mix[:,None,None])*command+mix[:,None,None]*slot_command,c.pursuer_speed)
            # A searcher retains its search trajectory; the remaining members
            # absorb the new angular slots and restore the closed formation.
            apply=self.active & (o['searcher']<0)[:,None]
            if self.method in ('gated_dynamic', 'hysteretic_dynamic', 'predictive_hysteretic'):
                apply &= self.gate_active[:, None]
            command=np.where(apply[:,:,None],slot_command,command)
        command=self.safety_filter(limit(command,c.pursuer_speed),o)
        command=np.where(self.active[...,None],command,0.0)
        return command,yaw_goal,altitude


    def commands(self, o, actions=None):
        self._update_reconfig_gate(o)
        view, obs, ids = self._compact(o)
        # Feedforward is ablated inside _core_commands before saturation;
        # search and safety retain the shared target-velocity estimate.
        aa = None if actions is None else actions[:, ids]
        u, yaw, z = view._core_commands(obs, aa)
        if self.staged.any():
            base = dict(obs, searcher=np.full(len(self.seeds), -1, int))
            bu, by, bz = view._core_commands(base, aa)
            mix = self.stage_mix
            u = np.where(self.staged[:, None, None], bu+mix[:, None, None]*(u-bu), u)
            yaw = np.where(self.staged[:, None], by+mix[:, None]*wrap(yaw-by), yaw)
            z = np.where(self.staged[:, None], bz+mix[:, None]*(z-bz), z)
        raw = u.copy()
        if self.method != 'no_response_guard' or not self.slot_control:
            # Withdrawn bodies are physical separation hazards, not teammates.
            guarded_obs = obs
            if view.departed_p.shape[1]:
                guarded_obs = dict(obs,
                    obstacles=np.concatenate([obs['obstacles'], view.departed_p], axis=1),
                    radii=np.concatenate([obs['radii'], np.full(view.departed_p.shape[:2], .6)], axis=1))
            u = response_guard(u, guarded_obs, 4., target_buffer=3.)
        full_u = np.zeros_like(self.p)
        full_raw = np.zeros_like(self.p)
        full_yaw = self.yaw.copy()
        full_z = np.full(self.z.shape, 3.)
        full_u[:, ids], full_raw[:, ids] = u, raw
        full_yaw[:, ids], full_z[:, ids] = yaw, z
        self.raw_command = full_raw
        self.guard_delta = full_u-full_raw
        return full_u, full_yaw, full_z

    def evader_command(self):
        # Inherited route/preview opponent sees only active pursuers. Restoring
        # the full array afterward keeps the physical environment unchanged.
        original = self.p
        ids = np.flatnonzero(self.active[0])
        self.p = original[:, ids]
        try:
            return super().evader_command()
        finally:
            self.p = original

    def withdraw(self, index, gain=.8, duration=1.):
        self.observe()
        # All methods share pre-fault nominal behaviour. Retained-slots freezes
        # the M-member reference at fault time, then leaves the vacated slot.
        self._reallocate_slots()
        previous = self.slot_angles.copy()
        mask = self.active.copy()
        mask[:, index] = False
        self.set_active(mask)
        if self.method != 'active_polar':
            self.enable_slot_control(gain=gain, duration=.1 if self.method == 'greedy' else duration)
            if self.method in ('gated_dynamic', 'hysteretic_dynamic', 'predictive_hysteretic'):
                self.gate_active = np.zeros(len(self.seeds), dtype=bool)
                self.gate_invalid_ticks = np.zeros(len(self.seeds), dtype=int)
                self.gate_valid_ticks = np.zeros(len(self.seeds), dtype=int)
            if self.method == 'retained_slots':
                self.slot_angles = previous
                self.slot_angles[~mask] = np.nan


class CornerWorld(AuditMixin, VariableCornerWorld):
    pass


class CorridorWorld(AuditMixin, VariableCorridorWorld):
    pass


def make_world(scene, seeds, task=None, scale=1.6, n_pursuers=4, method='dynamic'):
    if method not in METHODS:
        raise ValueError(method)
    task = task or Task(scene=scene)
    cls = {'corner': CornerWorld, 'corridor': CorridorWorld}[scene]
    world = cls(seeds, task, scale, n_pursuers)
    world.method = method
    world.configure(slice(None), 'staged_brake')
    return world
