"""Independent map/opponent protocol; inherited pursuer methods unchanged."""
from dataclasses import replace
import numpy as np

from experiments.c150_native_search.task import Task, visibility3
from experiments.c190_joint_moving_recovery.world import JointMovingWorld
from experiments.c191_route_validity.sampling import sample_valid
from experiments.c191_route_validity.opponent import RouteGuide


class ValidRouteWorld(JointMovingWorld):
    def __init__(self, seeds, *args, map_protocol='valid', opponent='routed', recovery='none', protocol_scale=1.6, **kwargs):
        super().__init__(seeds, *args, variant=recovery if recovery.startswith('moving_joint') else 'moving_joint', **kwargs)
        if not recovery.startswith('moving_joint'):
            self.recovery = recovery
            self.recovery_planner = None
        self.requested_recovery = recovery
        self.map_protocol, self.opponent = map_protocol, opponent
        self.validity_rejections = np.zeros(len(seeds), int)
        self.pair_clearance = np.full(len(seeds), np.nan)
        self.anchor_clearance = np.full(len(seeds), np.nan)
        scale = float(protocol_scale)
        if map_protocol=='valid':
            for i, seed in enumerate(seeds):
                rng = np.random.default_rng(np.random.SeedSequence([int(seed), 191, self.layout_split=='test']))
                rotation = self._scene_rotation(i)
                for _ in range(200):
                    local, radii, rejects, pair, anchor = sample_valid(rng, self.n_obstacles, self.shape_profile, self.layout_profile, scale)
                    self.validity_rejections[i] += rejects
                    obstacles = self.initial_target[i]+(local@rotation.T)*scale
                    radii = radii*scale
                    visible = visibility3(self.p[i:i+1], self.z[i:i+1], self.target[i:i+1], self.tz[i:i+1],
                        self.yaw[i:i+1], obstacles[None], radii[None], self.cfg)
                    clearance = (np.linalg.norm(self.p[i, :, None]-obstacles, axis=-1)-radii).min()
                    if visible.any() and clearance>.5:
                        self.obstacles[i], self.radii[i] = obstacles, radii
                        self.pair_clearance[i], self.anchor_clearance[i] = pair, anchor
                        break
                    self.validity_rejections[i] += 1
                else:
                    raise RuntimeError(f'C191 initial legality exhausted at seed {seed}')
            self._validate_initial_layout()
            from experiments.c179_random_map_generalization.world import capture_ring_feasible
            self.initial_ring_sample_found = np.array([capture_ring_feasible(o,r,t)
                for o,r,t in zip(self.obstacles,self.radii,self.target)])
            self.map_rejections = self.validity_rejections.copy()
        self.route_guide = RouteGuide(self) if opponent=='routed' else None
        self.last_evader_command = np.zeros_like(self.target)
        self.opponent_clock = None

    def evader_command(self):
        if self.opponent_clock==self.time:
            return self.last_evader_command.copy()
        command = self.route_guide.command(self) if self.route_guide is not None else super().evader_command()
        self.last_evader_command = command.copy()
        self.opponent_clock = self.time
        return command

    def branch_metadata(self):
        # JointMovingWorld metadata assumes a joint planner; use the parent
        # C180 metadata so none/routed_transit remain valid comparator arms.
        from experiments.c180_random_map_variable.world import RandomVariableCornerWorld
        meta = RandomVariableCornerWorld.branch_metadata(self)
        meta.update(recovery=self.requested_recovery, map_protocol=self.map_protocol, opponent=self.opponent,
            validity_rejections=self.validity_rejections.tolist(),
            route_guide_calls=self.route_guide.calls.tolist() if self.route_guide else [],
            route_guide_no_plan=self.route_guide.no_plan.tolist() if self.route_guide else [],
            validity_scope='all obstacle pairs and both hidden route branches; no capture-success screening')
        if hasattr(self.recovery_planner, 'joint_entries'):
            meta.update(joint_entries=self.recovery_planner.joint_entries.tolist(),
                        joint_plan_calls=self.recovery_planner.joint_calls.tolist(),
                        joint_no_plan=self.recovery_planner.joint_no_plan.tolist())
        return meta


def make_world(scene, seeds, task=None, scale=1.6, n_pursuers=4, method='active_polar',
               n_obstacles=4, shape_profile='cylinder', layout_profile='distributed',
               initial_ring_screen=False, recovery='none', map_protocol='valid', opponent='routed'):
    if scene not in ('random_train', 'random_test') or map_protocol not in ('legacy', 'valid') or opponent not in ('local', 'routed'):
        raise ValueError('unregistered C191 protocol')
    if initial_ring_screen:
        raise ValueError('C191 does not allow initial capture-ring screening')
    task = Task(scene='corner') if task is None else replace(task, scene='corner')
    world = ValidRouteWorld(seeds, task, scale, n_pursuers=n_pursuers, split='train' if scene=='random_train' else 'test',
        n_obstacles=n_obstacles, shape_profile=shape_profile, layout_profile=layout_profile,
        initial_ring_screen=False, recovery=recovery, map_protocol=map_protocol, opponent=opponent, protocol_scale=scale)
    world.configure(slice(None), 'staged_brake')
    world.method = method
    return world
