"""Global path guidance with the original bounded-heading velocity preview."""
import numpy as np

from experiments.c145_feasibility_reset.planar import norm, unit, limit, wrap
from experiments.c145_feasibility_reset.preview_evader import PreviewWorld
from experiments.c184_route_recovery.planner import VisibilityRouter, visible_edges


def remaining_cost(points, start, path):
    chain = np.r_[start[None], path]
    segment = np.diff(chain, axis=0)
    length = np.linalg.norm(segment, axis=1)
    denominator = np.maximum(length**2, 1e-10)
    alpha = np.clip(((points[:, None]-chain[:-1])*segment).sum(-1)/denominator, 0, 1)
    closest = chain[:-1]+alpha[..., None]*segment
    remaining = np.cumsum(length[::-1])[::-1][None]-alpha*length
    lateral = np.linalg.norm(points[:, None]-closest, axis=-1)
    nearest = np.argmin(lateral, axis=1)
    return (remaining+1.8*lateral)[np.arange(len(points)), nearest]


class RouteGuide:
    def __init__(self, world):
        self.routers = [VisibilityRouter(o, r+1., goals, bound=48.5, samples=32)
                        for o, r, goals in zip(world.obstacles, world.radii, world.routes)]
        self.paths = [None]*len(world.seeds)
        self.goal_index = np.full(len(world.seeds), -1, int)
        self.next_plan = np.zeros(len(world.seeds))
        self.calls = np.zeros(len(world.seeds), int)
        self.no_plan = np.zeros(len(world.seeds), int)

    def command(self, world):
        w = world
        before = w.route_index.copy()
        ids = np.arange(len(w.seeds))
        goal = w.routes[ids, w.route_index]
        reached = norm(goal-w.target)<2.
        w.route_done |= (w.route_index==2)&reached
        w.route_index = np.minimum(w.route_index+reached, 2)
        switched = (before==0)&(w.route_index>=1)
        w.branch_switch_tick[switched] = round(w.time*10)
        w.branch_switch_hidden[switched] = ~w.measured[switched]
        w.branch_reached |= w.route_index>=1
        old = w.heading.copy()
        free, free_heading = None, None
        if w.route_done.any():
            original = w.p
            try:
                w.p = original[:, np.flatnonzero(w.active[0])]
                free = PreviewWorld.evader_command(w)
                free_heading = w.heading.copy()
            finally:
                w.p = original
                w.heading = old.copy()
        output = np.zeros_like(w.target)
        for i in range(len(w.seeds)):
            if w.route_done[i]:
                output[i] = free[i]; w.heading[i] = free_heading[i]
                continue
            index = w.route_index[i]
            if w.time>=self.next_plan[i]-1e-8 or index!=self.goal_index[i]:
                self.calls[i] += 1
                _, paths = self.routers[i].paths_from(w.target[i])
                self.paths[i] = paths[index]
                self.goal_index[i] = index
                self.next_plan[i] = w.time+.5
                if self.paths[i] is None:
                    self.no_plan[i] += 1
            path = self.paths[i]
            if path is None:
                # Preserve a logged no-plan failure; use the local objective
                # with the same physical preview, never teleport or reset.
                path = w.routes[i, index][None]
            while len(path)>1 and np.linalg.norm(path[0]-w.target[i])<.65:
                if not visible_edges(w.target[i, None], path[1:2], w.obstacles[i], w.radii[i]+.8)[0, 0]:
                    break
                path = path[1:]
            self.paths[i] = path
            away = w.target[i]-w.p[i, w.active[i]]
            direction = unit(path[0]-w.target[i])+.2*(unit(away)/np.maximum(norm(away), 1)[:, None]).sum(0)
            desired = np.arctan2(direction[1], direction[0])
            heading = wrap(old[i]+np.clip(wrap(desired-old[i]), -.08, .08))
            headings = np.repeat(np.r_[heading, old[i]+np.array([-.08, -.04, 0., .04, .08])], 5)
            scales = np.tile([1., .75, .5, .25, 0.], 6)
            commands = 3.2*scales[:, None]*np.c_[np.cos(headings), np.sin(headings)]
            positions = np.repeat(w.target[i, None], 30, axis=0)
            velocities = np.repeat(w.tv[i, None], 30, axis=0)
            clearance = np.full(30, np.inf)
            for _ in range(20):
                velocities += .1*limit(2*(commands-velocities), 4.)
                positions += .1*velocities
                gap = (norm(positions[:, None]-w.obstacles[i])-w.radii[i]-.8).min(1)
                clearance = np.minimum(clearance, np.minimum(gap, (49.-np.abs(positions)).min(1)))
            score = remaining_cost(positions, w.target[i], path)
            score += .02*np.sum((commands-w.tv[i])**2, axis=1)
            score += .01*np.abs(wrap(headings-desired))
            score += np.where(clearance>=0, 0, 1e5+1e4*np.maximum(-clearance, 0))
            best = int(np.argmin(score))
            output[i] = commands[best]
            w.heading[i] = wrap(headings[best])
        return output
