"""Bounded forecast radial-spread correction, before inherited safety layers.

Coefficients are frozen from C211's descriptive fit on C210 recent24 traces.
No hidden state or future executed command is used. A held-current-command
forecast is not an invariant-set, safety, or native capture guarantee.
"""
import numpy as np
from experiments.c145_feasibility_reset.planar import norm, unit, limit
from experiments.c200_feedforward_budget.controller import allocate

DT = .1
STEPS = 5
RESPONSE_B = .49424477839019965
RESPONSE_ALPHA = .7208355562945309
SPREAD_LIMIT = 2.
MAX_CORRECTION = .6  # Existing residual-action velocity magnitude, not tuned.


def forecast(p, v, acceleration, command):
    """Trapezoidal position rollout and isotropic position/command Jacobian."""
    p = np.asarray(p, float).copy(); v = np.asarray(v, float).copy()
    a = np.asarray(acceleration, float).copy()
    cv = ca = jacobian = 0.
    for _ in range(STEPS):
        a = RESPONSE_B * (command-v) + RESPONSE_ALPHA*a
        next_v = v+DT*a
        p += .5*DT*(v+next_v)
        v = next_v
        ca = RESPONSE_B*(1-cv) + RESPONSE_ALPHA*ca
        next_cv = cv+DT*ca
        jacobian += .5*DT*(cv+next_cv)
        cv = next_cv
    return p, v, jacobian


def spread_filter(command, o, speed=4., diagnostics=False):
    """Contract predicted extreme radii while preserving predicted bearings.

The extra correction is capped at 0.6 m/s then shortened to fit the speed
ball. These two scalings only shorten the radial segment towards the chosen
2 m interval in this held-command model. Downstream safety/route changes and
target manoeuvres are outside that nominal statement.
"""
    command = np.asarray(command, float)
    p = o['p']; target = o['target']
    near = np.clip((12-norm(p-target[:, None]).max(-1))/4, 0, 1)
    enabled = o['measured'] & (o['searcher'] < 0) & o.get('acceleration_valid', np.zeros(len(p), bool)) & (near > 0)
    base = limit(command, speed)
    predicted, _, jac = forecast(p, o['v'], o.get('own_acceleration', np.zeros_like(p)), base)
    center = target+DT*STEPS*o['target_v']
    delta = predicted-center[:, None]
    radii = norm(delta); er = unit(delta)
    middle = .5*(radii.max(-1)+radii.min(-1))
    desired = np.clip(radii, middle[:, None]-SPREAD_LIMIT/2, middle[:, None]+SPREAD_LIMIT/2)
    correction = limit(((desired-radii)/jac)[..., None]*er, MAX_CORRECTION)*near[:, None, None]
    proposal, alpha = allocate(base, correction, speed)
    enabled &= np.ptp(radii, axis=-1) > SPREAD_LIMIT
    result = np.where(enabled[:, None, None], proposal, command)
    if not diagnostics: return result
    adjusted = predicted + jac*(proposal-base)
    return result, dict(enabled=enabled, before_spread=np.ptp(radii, axis=-1),
        after_spread=np.ptp(norm(adjusted-center[:, None]), axis=-1),
        correction_mps=norm(proposal-base), alpha=alpha, jacobian=jac)
