"""Apply the frozen C200 allocation only after task withdrawal.

The compact view carries only surviving formation members. Retired bodies
remain in the same inherited safety filter. No new prediction or sensor input.
"""
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c200_feedforward_budget.world import BudgetRouteWorld, make_world as old_world


class PostFaultBudgetWorld(BudgetRouteWorld):
    def _core_commands(self, o, actions=None):
        departed = getattr(self, 'departed_p', None)
        if departed is not None and departed.shape[1]:
            return super()._core_commands(o, actions)
        return AuditMixin._core_commands(self, o, actions)


def make_world(*args, recovery='routed_transit', **kwargs):
    if recovery not in ('postfault_budget', 'budget_sector'):
        return old_world(*args, recovery=recovery, **kwargs)
    parent = 'sector_bypass' if recovery == 'budget_sector' else 'routed_transit'
    world = old_world(*args, recovery=parent, **kwargs)
    assert world.n_pursuers in (4, 5)
    world.__class__ = PostFaultBudgetWorld
    world.requested_recovery = recovery
    return world
