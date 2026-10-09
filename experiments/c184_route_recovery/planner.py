"""Obstacle-aware formation recovery using public observations only.

A candidate planner, not a safety certificate. The caller retains the existing
active-only safety projection and response guard, and the original judge.
"""
from __future__ import annotations

import heapq
import itertools
import numpy as np


def visible_edges(a, b, centers, radii):
    """All segment a[i]--b[j] clear of closed circular envelopes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = b[None] - a[:, None]
    denom = np.maximum(np.sum(delta * delta, axis=-1), 1e-12)
    valid = np.ones(denom.shape, bool)
    for c, r in zip(centers, radii):
        alpha = np.clip(np.sum((c-a)[:, None]*delta, axis=-1)/denom, 0, 1)
        closest = a[:, None] + alpha[..., None]*delta
        valid &= np.sum((closest-c)**2, axis=-1) >= r*r-1e-8
    return valid


class VisibilityRouter:
    """Circumscribed obstacle polygons, exact segment checks, Dijkstra."""
    def __init__(self, centers, radii, destinations, bound=48.5, samples=20):
        self.centers = np.asarray(centers, float).reshape(-1, 2)
        self.radii = np.asarray(radii, float)
        angles = np.arange(samples)*2*np.pi/samples
        direction = np.stack([np.cos(angles), np.sin(angles)], -1)
        vertices = self.centers[:, None] + (self.radii/np.cos(np.pi/samples)+.04)[:, None, None]*direction
        vertices = vertices.reshape(-1, 2)
        clear = (np.linalg.norm(vertices[:, None]-self.centers, axis=-1)-self.radii).min(-1)
        vertices = vertices[(clear > 1e-5) & (np.abs(vertices) < bound).all(-1)]
        self.destinations = np.asarray(destinations, float).reshape(-1, 2)
        self.nodes = np.r_[self.destinations, vertices]
        edges = visible_edges(self.nodes, self.nodes, self.centers, self.radii)
        distance = np.linalg.norm(self.nodes[:, None]-self.nodes[None], axis=-1)
        self.weights = np.where(edges & ~np.eye(len(edges), dtype=bool), distance, np.inf)

    def paths_from(self, start):
        start = np.asarray(start, float)
        delta = self.nodes-start
        possible = np.ones(len(self.nodes), bool)
        for c, r in zip(self.centers, self.radii):
            if np.linalg.norm(start-c) < r:
                # The physical state can be inside the planner's conservative
                # inflation. Allow only monotone outward escape, never a chord.
                possible &= ((delta*(start-c)).sum(-1) >= -1e-8)
                possible &= np.linalg.norm(self.nodes-c, axis=-1) >= r
            else:
                possible &= visible_edges(start[None], self.nodes, c[None], [r])[0]
        distance = np.where(possible, np.linalg.norm(delta, axis=-1), np.inf)
        previous = np.full(len(self.nodes), -1, int)
        queue = [(float(distance[i]), i) for i in np.flatnonzero(possible)]
        heapq.heapify(queue)
        while queue:
            current, j = heapq.heappop(queue)
            if current > distance[j]+1e-9:
                continue
            candidate = current + self.weights[j]
            for k in np.flatnonzero(candidate < distance-1e-9):
                distance[k] = candidate[k]
                previous[k] = j
                heapq.heappush(queue, (float(candidate[k]), int(k)))
        paths = []
        for j in range(len(self.destinations)):
            if not np.isfinite(distance[j]):
                paths.append(None)
                continue
            ids, k = [j], previous[j]
            while k >= 0:
                ids.append(k)
                k = previous[k]
            paths.append(self.nodes[ids[::-1]].copy())
        return distance[:len(self.destinations)], paths


def formation_bank():
    """Interior of the unchanged 3-agent 4..8 m, 90..150 deg judge."""
    gaps = [(a, b) for a, b in itertools.product((95, 105, 115, 125, 135, 145), repeat=2)
            if 95 <= 360-a-b <= 145]
    radii = np.asarray([r for r in itertools.product((4.3, 5.15, 6., 6.85, 7.7), repeat=3)
                        if max(r)-min(r) <= 1.8+1e-9])
    base = np.deg2rad([[0, a, a+b] for a,b in gaps])
    angles = base[:, None, :] + np.arange(72)[None, :, None]*2*np.pi/72
    directions = np.stack([np.cos(angles), np.sin(angles)], -1)
    return (directions[:, :, None] * radii[None, None, :, :, None]).reshape(-1,3,2)


OFFSETS = formation_bank()


def plan_formation(p, center, obstacles, radii, previous=None, margin=1.4):
    """Pick clear non-equilateral slots by routed travel cost; no map deletion."""
    p, center = np.asarray(p), np.asarray(center)
    if p.shape != (3,2):
        return None
    obs = np.asarray(obstacles).reshape(-1,2)
    radius = np.asarray(radii)+margin
    order = np.argsort(np.arctan2((p-center)[:,1], (p-center)[:,0]))
    goals = center + OFFSETS
    clear = (np.linalg.norm(goals[...,None,:]-obs, axis=-1)-radius).min((-1,-2))
    legal = (clear >= 0) & (np.abs(goals) < 48.5).all((-1,-2))
    choices = np.flatnonzero(legal)
    if not len(choices):
        return None
    offsets = OFFSETS[choices]
    cost = np.linalg.norm(goals[choices]-p[order], axis=-1).sum(-1)
    cost += .15*np.sum((np.linalg.norm(offsets,axis=-1)-6.)**2,axis=-1)
    cost -= .5*np.minimum(clear[choices], 2.)
    if previous is not None:
        cost += .6*np.linalg.norm(offsets-previous[order],axis=-1).sum(-1)
    # Route a diverse shortlist rather than all sampled formations.
    shortlist=[]
    for j in np.argsort(cost):
        if all(np.linalg.norm(offsets[j]-offsets[k]) > 2.0 for k in shortlist):
            shortlist.append(int(j))
            if len(shortlist) == 8:
                break
    selected = choices[shortlist]
    candidate_goals = goals[selected]
    router = VisibilityRouter(np.r_[obs,center[None]], np.r_[radius,3.8], candidate_goals.reshape(-1,2))
    routes = [router.paths_from(start) for start in p[order]]
    travel = np.stack([r[0].reshape(-1,3)[:,i] for i,r in enumerate(routes)],axis=1)
    routed_cost = travel.sum(-1) + .2*np.asarray(cost)[shortlist]
    if not np.isfinite(routed_cost).any():
        return None
    best=int(np.argmin(routed_cost)); paths=[None]*3
    local=np.zeros((3,2))
    for slot,agent in enumerate(order):
        paths[agent]=routes[slot][1][3*best+slot]
        local[agent]=candidate_goals[best,slot]-center
    return dict(paths=paths, offsets=local, cost=float(routed_cost[best]))


class RecoveryPlanner:
    """Triggered only after observed geometry loss with near-obstacle stagnation."""
    def __init__(self, batch, agents):
        self.on=np.zeros(batch,bool)
        self.invalid_time=np.zeros(batch)
        self.valid_time=np.zeros(batch)
        self.entries=np.zeros(batch,int)
        self.no_plan=np.zeros(batch,int)
        self.plan_calls=np.zeros(batch,int)
        self.paths=[None]*batch
        self.offsets=[None]*batch
        self.next_plan=np.zeros(batch)
        self.clock=None
        self.last_output=None
        self.agents=agents

    def propose(self,o,base,time):
        if self.clock == time and self.last_output is not None:
            return self.last_output
        dt=0. if self.clock is None else max(0.,time-self.clock)
        self.clock=time
        out=np.asarray(base).copy(); changed=np.zeros(len(base),bool)
        for i in range(len(base)):
            ids=np.flatnonzero(o['active'][i])
            if len(ids)!=3 or (self.agents>3 and len(ids)==self.agents) or not o['measured'][i] or np.linalg.norm(o['target_v'][i])>2.5:
                self.on[i]=False;self.invalid_time[i]=0;self.valid_time[i]=0
                continue
            p=o['p'][i,ids];center=o['target'][i]
            delta=p-center;rr=np.linalg.norm(delta,axis=-1)
            angle=np.sort(np.arctan2(delta[:,1],delta[:,0]));gap=np.diff(np.r_[angle,angle[0]+2*np.pi])
            valid=rr.min()>=4.2 and rr.max()<=7.8 and np.ptp(rr)<=1.8 and gap.min()>=np.deg2rad(95) and gap.max()<=np.deg2rad(145)
            retired=o['p'][i,~o['active'][i]]
            obs=np.r_[o['obstacles'][i],retired]
            radii=np.r_[o['radii'][i],np.full(len(retired),.6)]
            clear=(np.linalg.norm(p[:,None]-obs,axis=-1)-radii).min(-1)
            stall=((np.linalg.norm(o['v'][i,ids],axis=-1)<.45)&(np.linalg.norm(base[i,ids],axis=-1)<.6)&(clear<2.5)).any()
            self.invalid_time[i]=self.invalid_time[i]+dt if not valid and stall else 0.
            self.valid_time[i]=self.valid_time[i]+dt if valid else 0.
            if not self.on[i] and self.invalid_time[i]>=1.0:
                self.on[i]=True;self.entries[i]+=1;self.next_plan[i]=time;self.offsets[i]=None
            if self.on[i] and self.valid_time[i]>=1.0:
                self.on[i]=False;self.paths[i]=None
            if not self.on[i]:continue
            if time>=self.next_plan[i]-1e-6:
                self.plan_calls[i]+=1
                plan=plan_formation(p,center,obs,radii,self.offsets[i])
                self.next_plan[i]=time+1.
                if plan is None:
                    self.no_plan[i]+=1;self.paths[i]=None
                else:
                    self.paths[i]=plan['paths'];self.offsets[i]=plan['offsets']
            if self.paths[i] is None:continue
            changed[i]=True
            for j,agent in enumerate(ids):
                path=self.paths[i][j]
                while len(path)>1 and np.linalg.norm(path[0]-p[j])<.65:
                    path=path[1:]
                self.paths[i][j]=path
                if len(path)==1:
                    goal=center+self.offsets[i][j]
                    vel=.9*(goal-p[j])+o['target_v'][i]
                else:
                    vel=1.1*(path[0]-p[j])
                out[i,agent]=vel*min(1.,4.0/max(np.linalg.norm(vel),1e-8))
        self.last_output=(out,changed)
        return self.last_output
