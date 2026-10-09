"""C210 controller with a bounded, causal forecast spread filter."""
import numpy as np
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c208_metric_budget.budget_core import MetricBudgetCore
from experiments.c209_coverage_handover.controller import team_weight
from experiments.c210_radial_damping.world import RadialDampedWorld
from experiments.c211_uniformity_handover.world import make_world as previous_world
from experiments.c212_predictive_spread.formation_core import PredictiveFormationCore


class PredictiveSpreadWorld(RadialDampedWorld):
    def commands(self, o, actions=None):
        previous = getattr(self, '_response_history', None)
        now = float(self.time)
        if previous is None or now != previous['time']:
            valid = previous is not None and abs(now-previous['time']-.1) < 1e-7
            accel = (o['v']-previous['v'])/.1 if valid else np.zeros_like(o['v'])
            self._response_history = dict(time=now, v=o['v'].copy(), acceleration=accel.copy(), valid=valid)
        state = self._response_history
        obs = dict(o, own_acceleration=state['acceleration'].copy(),
                   acceleration_valid=np.full(len(o['p']), state['valid'], bool))
        return super().commands(obs, actions)

    def _compact(self, o):
        view, obs, ids = super()._compact(o)
        if 'own_acceleration' in o: obs['own_acceleration'] = o['own_acceleration'][:, ids].copy()
        return view, obs, ids

    def _core_commands(self, o, actions=None):
        departed = getattr(self, 'departed_p', None)
        if departed is None or not departed.shape[1]:
            return AuditMixin._core_commands(self, o, actions)
        weight = team_weight(o)
        if (weight == 1).all(): return MetricBudgetCore._core_commands(self, o, actions)
        old = PredictiveFormationCore._core_commands(self, o, actions)
        if (weight == 0).all(): return old
        new = MetricBudgetCore._core_commands(self, o, actions)
        return old[0]+weight[:, None, None]*(new[0]-old[0]), old[1], old[2]


def make_world(*args, recovery='routed_transit', **kwargs):
    if recovery != 'predictive_spread_sector': return previous_world(*args, recovery=recovery, **kwargs)
    w = previous_world(*args, recovery='radial_damped_sector', **kwargs)
    w.__class__ = PredictiveSpreadWorld
    w.requested_recovery = recovery
    original = w.recovery_planner.propose
    def recorded(o, base, time):
        result = original(o, base, time)
        w.recovery_planner.audit_last['own_acceleration'] = o['own_acceleration'].copy()
        w.recovery_planner.audit_last['acceleration_valid'] = o['acceleration_valid'].copy()
        return result
    w.recovery_planner.propose = recorded
    return w
