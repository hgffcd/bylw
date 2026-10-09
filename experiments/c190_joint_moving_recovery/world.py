"""Isolated C190 wrapper; C180/C189 controller and source remain intact."""
from dataclasses import replace

from experiments.c150_native_search.task import Task
from experiments.c180_random_map_variable.world import RandomVariableCornerWorld, make_world as original_world
from experiments.c190_joint_moving_recovery.planner import JointMovingPlanner


class JointMovingWorld(RandomVariableCornerWorld):
    def __init__(self, seeds, *args, **kwargs):
        self.variant = kwargs.pop('variant', 'moving_joint')
        kwargs['recovery'] = 'routed_transit'
        super().__init__(seeds, *args, **kwargs)
        if self.variant == 'moving_joint_v2':
            from experiments.c190_joint_moving_recovery.planner_v2 import FlexibleMovingPlanner
            self.recovery_planner = FlexibleMovingPlanner(len(seeds), self.n_pursuers)
        else:
            self.recovery_planner = JointMovingPlanner(len(seeds), self.n_pursuers)

    def branch_metadata(self):
        meta = super().branch_metadata()
        planner = self.recovery_planner
        meta.update(recovery=self.variant, joint_entries=planner.joint_entries.tolist(),
                    joint_plan_calls=planner.joint_calls.tolist(), joint_no_plan=planner.joint_no_plan.tolist(),
                    recovery_delta_time_constant_s=.45)
        return meta


def make_world(scene, seeds, task=None, scale=1.6, n_pursuers=4,
               method='active_polar', n_obstacles=4, shape_profile='cylinder',
               layout_profile='distributed', initial_ring_screen=False, recovery='none'):
    if recovery not in ('moving_joint', 'moving_joint_v2'):
        return original_world(scene, seeds, task, scale, n_pursuers, method, n_obstacles,
                              shape_profile, layout_profile, initial_ring_screen, recovery)
    if scene not in ('random_train', 'random_test'):
        raise ValueError(scene)
    task = Task(scene='corner') if task is None else replace(task, scene='corner')
    world = JointMovingWorld(seeds, task, scale, n_pursuers=n_pursuers,
        split='train' if scene=='random_train' else 'test', n_obstacles=n_obstacles,
        shape_profile=shape_profile, layout_profile=layout_profile,
        initial_ring_screen=initial_ring_screen, variant=recovery)
    world.configure(slice(None), 'staged_brake')
    world.method = method
    return world
