"""Joint routed recovery for a moving observed target; development candidate.

All members receive one jointly selected ring. Original physical safety and
response projections remain in the caller. No hidden target/route state.
"""
import numpy as np

from experiments.c184_route_recovery.planner import VisibilityRouter
from experiments.c187_transit_recovery.planner import TransitRecoveryPlanner


def plan_joint(p, center, obstacles, radii, previous=None):
    p, center = np.asarray(p, float), np.asarray(center, float)
    n = len(p)
    if n not in (3, 4):
        raise ValueError('joint plan supports three or four active members')
    delta = p-center
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    order = np.argsort(angle)
    nominal = np.arange(n)*2*np.pi/n
    phase = np.mean(np.unwrap(angle[order])-nominal)
    phases = phase+np.linspace(-np.pi/n, np.pi/n, 25)
    direction = np.stack([np.cos(phases[:, None]+nominal),
                          np.sin(phases[:, None]+nominal)], axis=-1)
    offsets = (direction[:, None]*np.array([5.2, 6., 6.8])[None, :, None, None]).reshape(-1, n, 2)
    goals = center+offsets
    obstacles = np.asarray(obstacles, float).reshape(-1, 2)
    envelopes = np.asarray(radii, float)+1.4
    clearance = (np.linalg.norm(goals[..., None, :]-obstacles, axis=-1)-envelopes).min((-1, -2))
    legal = (clearance>=0) & (np.abs(goals)<48.5).all((-1, -2))
    choices = np.flatnonzero(legal)
    if not len(choices):
        return None
    distance = np.linalg.norm(goals[choices]-p[order], axis=-1)
    cost = distance.max(1)+.2*distance.sum(1)-.2*np.minimum(clearance[choices], 2)
    if previous is not None:
        cost += .5*np.linalg.norm(offsets[choices]-previous[order], axis=-1).max(1)
    selected = []
    for j in np.argsort(cost):
        if all(np.linalg.norm(offsets[choices[j]]-offsets[choices[k]])>2 for k in selected):
            selected.append(int(j))
            if len(selected)==6:
                break
    selected_goals = goals[choices[selected]]
    router = VisibilityRouter(np.r_[obstacles, center[None]],
                              np.r_[envelopes, 3.4], selected_goals.reshape(-1, 2))
    routes = [router.paths_from(start) for start in p[order]]
    travel = np.stack([r[0].reshape(-1, n)[:, j] for j, r in enumerate(routes)], axis=1)
    score = travel.max(1)+.2*travel.sum(1)+.15*cost[selected]
    if not np.isfinite(score).any():
        return None
    best = int(np.argmin(score))
    paths, local = [None]*n, np.zeros((n, 2))
    for slot, agent in enumerate(order):
        paths[agent] = routes[slot][1][n*best+slot]
        local[agent] = selected_goals[best, slot]-center
    return dict(paths=paths, offsets=local, cost=float(score[best]))


class JointMovingPlanner(TransitRecoveryPlanner):
    def __init__(self, batch, agents):
        super().__init__(batch, agents)
        self.joint_on = np.zeros(batch, bool)
        self.joint_changed = np.zeros(batch, bool)
        self.blend_changed = np.zeros(batch, bool)
        self.joint_entries = np.zeros(batch, int)
        self.joint_calls = np.zeros(batch, int)
        self.joint_no_plan = np.zeros(batch, int)
        self.stall_time = np.zeros(batch)
        self.joint_valid_time = np.zeros(batch)
        self.joint_next = np.zeros(batch)
        self.joint_paths = [None]*batch
        self.joint_offsets = [None]*batch
        self.smooth_delta = np.zeros((batch, agents, 2))
        self.previous_active = None
        self.joint_clock = None
        self.joint_output = None

    def propose(self, o, base, time):
        if self.joint_clock == time and self.joint_output is not None:
            return self.joint_output
        dt = 0. if self.joint_clock is None else max(0., time-self.joint_clock)
        self.joint_clock = time
        proposal, changed = super().propose(o, base, time)
        proposal, changed = proposal.copy(), changed.copy()
        self.joint_changed[:] = False
        if self.previous_active is not None:
            reset = (o['active'] != self.previous_active).any(1)
            self.smooth_delta[reset] = 0
            self.joint_on[reset] = False
            self.stall_time[reset] = 0
            for i in np.flatnonzero(reset):
                self.joint_paths[i] = None
                self.joint_offsets[i] = None
        self.previous_active = o['active'].copy()
        for i in range(len(base)):
            ids = np.flatnonzero(o['active'][i])
            eligible = len(ids) in (3, 4) and len(ids)<self.agents and bool(o['measured'][i])
            if not eligible:
                self.joint_on[i] = False
                self.stall_time[i] = 0
                self.joint_valid_time[i] = 0
                continue
            p, center = o['p'][i, ids], o['target'][i]
            d = p-center
            rr = np.linalg.norm(d, axis=-1)
            angles = np.sort(np.arctan2(d[:, 1], d[:, 0]))
            gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
            nominal = 2*np.pi/len(ids)
            valid = (rr.min()>=4.3 and rr.max()<=7.5 and np.ptp(rr)<=1.5
                     and gaps.min()>=.8*nominal and gaps.max()<=1.2*nominal)
            retired = o['p'][i, ~o['active'][i]]
            obstacles = np.r_[o['obstacles'][i], retired]
            radii = np.r_[o['radii'][i], np.full(len(retired), .6)]
            clearance = (np.linalg.norm(p[:, None]-obstacles, axis=-1)-radii).min(-1)
            stalled = ((np.linalg.norm(base[i, ids], axis=-1)<.6)
                       & (np.linalg.norm(o['v'][i, ids], axis=-1)<.6) & (clearance<3.)).any()
            trigger = not valid and stalled and np.linalg.norm(o['target_v'][i])>1.
            self.stall_time[i] = self.stall_time[i]+dt if trigger else 0.
            self.joint_valid_time[i] = self.joint_valid_time[i]+dt if valid else 0.
            if not self.joint_on[i] and self.stall_time[i]>=.5:
                self.joint_on[i] = True
                self.joint_entries[i] += 1
                self.joint_next[i] = time
                self.joint_offsets[i] = None
            # Speed does not toggle the recovery off; geometry controls handover.
            if self.joint_valid_time[i]>=.8 or self.precision_changed[i]:
                self.joint_on[i] = False
            if not self.joint_on[i]:
                continue
            if time>=self.joint_next[i]-1e-6:
                self.joint_calls[i] += 1
                plan = plan_joint(p, center, obstacles, radii, self.joint_offsets[i])
                self.joint_next[i] = time+.6
                if plan is None:
                    self.joint_no_plan[i] += 1
                    self.joint_paths[i] = None
                else:
                    self.joint_paths[i] = plan['paths']
                    self.joint_offsets[i] = plan['offsets']
            if self.joint_paths[i] is None:
                continue
            changed[i] = True
            self.joint_changed[i] = True
            for j, agent in enumerate(ids):
                path = self.joint_paths[i][j]
                while len(path)>1 and np.linalg.norm(path[0]-p[j])<.55:
                    path = path[1:]
                self.joint_paths[i][j] = path
                if len(path)==1:
                    goal = center+self.joint_offsets[i][j]
                    velocity = o['target_v'][i]+.85*(goal-p[j])
                else:
                    delta = path[0]-p[j]
                    velocity = 4.*delta/max(np.linalg.norm(delta), 1e-9)
                proposal[i, agent] = velocity*min(1., 4./max(np.linalg.norm(velocity), 1e-9))
        desired_delta = np.where(changed[:, None, None], proposal-base, 0.)
        self.smooth_delta += (1.-np.exp(-dt/.45))*(desired_delta-self.smooth_delta)
        # A sensor loss hands control back to the unchanged search mechanism.
        self.smooth_delta[~o['measured']] = 0
        self.smooth_delta *= o['active'][..., None]
        self.blend_changed = np.linalg.norm(self.smooth_delta, axis=(1, 2))>1e-4
        self.joint_output = base+self.smooth_delta, self.blend_changed.copy()
        return self.joint_output
