"""C104 joint short-path target guard. Model checks are not physics guarantees."""
from dataclasses import dataclass
from itertools import product
import math

import torch


@dataclass
class TargetPathGuardResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    selected_index: torch.Tensor
    nominal_minimum_m: torch.Tensor
    selected_minimum_m: torch.Tensor
    selected_infeasible: torch.Tensor
    maximum_agent_correction_mps: torch.Tensor


def observed_target_acceleration(current_velocity, previous_velocity, visible,
                                 previous_visible, *, dt=.1, maximum_acceleration=4.):
    """Use only consecutive shared observations; reacquisition has no history."""
    if (current_velocity.ndim != 2 or current_velocity.shape[-1] != 2
            or previous_velocity.shape != current_velocity.shape
            or visible.shape != current_velocity.shape[:1] or previous_visible.shape != visible.shape
            or visible.dtype != torch.bool or previous_visible.dtype != torch.bool):
        raise ValueError('observed acceleration state/mask shape mismatch')
    if dt <= 0 or maximum_acceleration <= 0:
        raise ValueError('observed acceleration scales must be positive')
    valid = (visible & previous_visible & torch.isfinite(current_velocity).all(dim=-1)
             & torch.isfinite(previous_velocity).all(dim=-1))
    estimate = (current_velocity-previous_velocity)/dt
    estimate = estimate*(maximum_acceleration/estimate.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
    return torch.where(valid[:, None], estimate, torch.zeros_like(estimate)), valid


def bounded_ca_target_path(target, velocity, acceleration, *, dt, steps, maximum_speed):
    """Constant observed acceleration scenario with per-step speed projection.

    This is a prediction assumption, not a bound on contact-driven target motion.
    The separate CV scenario is retained by the caller.
    """
    if target.shape != velocity.shape or acceleration.shape != velocity.shape:
        raise ValueError('target CA state shape mismatch')
    if min(dt, maximum_speed) <= 0 or steps < 1:
        raise ValueError('target CA scales must be positive')
    p, v = target.clone(), velocity.clone()
    positions = []
    for _ in range(steps):
        v = v+dt*acceleration
        v = v*(maximum_speed/v.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
        p = p+dt*v
        positions.append(p)
    return torch.stack(positions, dim=1)


def constant_command_path(position, velocity, commands, *, dt=.01, steps=50,
                          kp=2., acceleration=4.):
    """Vectorized solution of semi-implicit, norm-capped proportional velocity.

    For a fixed command the velocity-error direction stays constant. Its norm
    first decreases by a*dt per step, then geometrically by (1-kp*dt).
    This exactly represents that planar discrete model, not the attitude loop.
    Shapes: state [B,A,2], commands [B,C,A,2], path [B,C,T,A,2].
    """
    if min(dt, kp, acceleration) <= 0 or not 0 < kp*dt < 1 or steps < 1:
        raise ValueError('invalid planar rollout parameters')
    if (position.ndim != 3 or position.shape[-1] != 2 or velocity.shape != position.shape
            or commands.ndim != 4 or commands.shape[0] != position.shape[0]
            or commands.shape[2:] != position.shape[1:]):
        raise ValueError('planar rollout shape mismatch')
    error_vector = commands-velocity[:, None]
    error = error_vector.norm(dim=-1)
    direction = error_vector/error[..., None].clamp_min(1e-12)
    decrement = acceleration*dt
    saturated_steps = torch.ceil(((error-acceleration/kp)/decrement).clamp_min(0))
    k = torch.arange(1, steps+1, dtype=commands.dtype, device=commands.device)[None, None, :, None]
    n = torch.minimum(k, saturated_steps[:, :, None])
    remaining = (k-saturated_steps[:, :, None]).clamp_min(0)
    saturated_sum = n*error[:, :, None]-.5*decrement*n*(n+1)
    q = 1-kp*dt
    tail_error = (error-decrement*saturated_steps).clamp_min(0)
    tail_sum = tail_error[:, :, None]*q*(1-torch.pow(q, remaining))/(1-q)
    displacement = (k[..., None]*dt*commands[:, :, None]
                    -dt*(saturated_sum+tail_sum)[..., None]*direction[:, :, None])
    return position[:, None, None]+displacement


def path_violations(path, target, target_velocity, obstacle_xy, obstacle_radius, *,
                    dt, target_separation, agent_separation, pursuer_radius,
                    obstacle_clearance, arena_half_extent, boundary_clearance,
                    target_acceleration=None, target_prediction_speed=4.):
    """Keep individual target/agent/obstacle/axis constraints distinct."""
    times = torch.arange(1, path.shape[2]+1, device=path.device, dtype=path.dtype)*dt
    target_path = target[:, None]+times[None, :, None]*target_velocity[:, None]
    target_min = (path-target_path[:, None, :, None]).norm(dim=-1).amin(dim=2)
    if target_acceleration is not None:
        ca_path = bounded_ca_target_path(target, target_velocity, target_acceleration,
            dt=dt, steps=path.shape[2], maximum_speed=target_prediction_speed)
        ca_min = (path-ca_path[:, None, :, None]).norm(dim=-1).amin(dim=2)
        # Keep three separate per-agent constraints for EACH scenario. A gain
        # under CA must not hide a new CV violation, or vice versa.
        target_min = torch.cat((target_min, ca_min), dim=-1)
    target_bad = (target_separation-target_min).clamp_min(0)
    pair_min = (path[..., [0, 0, 1], :]-path[..., [1, 2, 2], :]).norm(dim=-1).amin(dim=2)
    pair_bad = (agent_separation-pair_min).clamp_min(0)
    boundary_bad = (path.abs()-(arena_half_extent-boundary_clearance)).clamp_min(0).amax(dim=2).flatten(2)
    if obstacle_xy.shape[1]:
        distance = (path[..., None, :]-obstacle_xy[:, None, None, None]).norm(dim=-1)
        obstacle_bad = (pursuer_radius+obstacle_clearance+obstacle_radius[:, None, None, None]-distance).clamp_min(0).amax(dim=2).flatten(2)
    else:
        obstacle_bad = path.new_zeros((*path.shape[:2], 0))
    return target_min, torch.cat((target_bad, pair_bad, boundary_bad, obstacle_bad), dim=-1)


def joint_target_path_guard(nominal_velocity, current_velocity, pursuer_xy,
                           target_xy, target_velocity, active, obstacle_xy, obstacle_radius, *,
                           maximum_speed=4., horizon=.5, dt=.01, velocity_kp=2., acceleration=4.,
                           target_separation=1.5, agent_separation=3.5, pursuer_radius=.5,
                           obstacle_clearance=1.5, arena_half_extent=50., boundary_clearance=1.,
                           target_acceleration=None, target_prediction_speed=4.,
                           margin_tiebreak=False, prediction_reserve_m=0.):
    """Retain or jointly repair final commands before physical execution.

    Each UAV can keep, brake to zero, move outward, or move in either tangent
    direction. All 5^3 joint commands are bounded by the unchanged speed limit.
    A repair must not increase ANY individual sampled safety violation versus
    nominal, and must strictly reduce total target-buffer violation. Among
    equivalent repairs use least command change. The opt-in C107 tie-break first
    maximizes the worst predicted target distance among equal-violation repairs.
    C109 optionally adds a prediction reserve to the target constraints and
    trigger. This does not change the actual task's target-separation limit.
    The reported selected_infeasible flag then refers to that larger prediction
    threshold. Every individual constraint remains non-worsening; a reserve
    is a prediction assumption, not a bound on model error.
    Infeasibility means only this
    candidate set/model failed; it is never labeled physically unavoidable.
    """
    b = pursuer_xy.shape[0]
    if (pursuer_xy.shape != (b, 3, 2) or nominal_velocity.shape != pursuer_xy.shape
            or current_velocity.shape != pursuer_xy.shape or target_xy.shape != (b, 2)
            or target_velocity.shape != (b, 2) or active.shape != (b,) or active.dtype != torch.bool):
        raise ValueError('target path guard requires three XY agents and a boolean batch mask')
    if (obstacle_xy.ndim != 3 or obstacle_xy.shape[0] != b or obstacle_xy.shape[-1] != 2
            or obstacle_radius.shape != obstacle_xy.shape[:2]):
        raise ValueError('target path guard obstacle shape mismatch')
    if min(maximum_speed, horizon, dt, velocity_kp, acceleration, target_separation,
           agent_separation, pursuer_radius, arena_half_extent) <= 0:
        raise ValueError('target path guard scales must be positive')
    if obstacle_clearance < 0 or not 0 <= boundary_clearance < arena_half_extent:
        raise ValueError('invalid target path guard clearances')
    if target_acceleration is not None and (target_acceleration.shape != (b, 2) or target_prediction_speed <= 0):
        raise ValueError('target acceleration scenario shape/scale mismatch')
    if not math.isfinite(prediction_reserve_m) or prediction_reserve_m < 0:
        raise ValueError('prediction reserve must be finite and non-negative')
    prediction_separation = target_separation + prediction_reserve_m
    steps = round(horizon/dt)
    if steps < 1 or not math.isclose(steps*dt, horizon, abs_tol=1e-8):
        raise ValueError('path horizon must divide the physical step')
    finite = torch.ones_like(active)
    for value in (nominal_velocity, current_velocity, pursuer_xy, target_xy,
                  target_velocity, obstacle_xy, obstacle_radius):
        finite &= torch.isfinite(value).reshape(b, -1).all(dim=1)
    if target_acceleration is not None:
        finite &= torch.isfinite(target_acceleration).all(dim=-1)
    valid = active & finite
    zeros = nominal_velocity.new_zeros(b)
    result = TargetPathGuardResult(nominal_velocity.clone(), valid, torch.zeros_like(valid),
        torch.zeros_like(active, dtype=torch.long), zeros.clone(), zeros.clone(),
        torch.zeros_like(valid), zeros.clone())
    if not valid.any():
        return result
    ids = valid.nonzero().squeeze(1)
    p, v, nominal = pursuer_xy[ids], current_velocity[ids], nominal_velocity[ids]
    target, tv, obs, radii = target_xy[ids], target_velocity[ids], obstacle_xy[ids], obstacle_radius[ids]
    if (nominal.norm(dim=-1) > maximum_speed+1e-5).any():
        raise ValueError('nominal command exceeds the unchanged speed limit')
    radial = p-target[:, None]
    norm = radial.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(radial); fallback[..., 0] = 1
    radial = torch.where(norm > 1e-9, radial/norm.clamp_min(1e-9), fallback)
    tangent = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)
    local = torch.stack((nominal, torch.zeros_like(nominal), maximum_speed*radial,
                         maximum_speed*tangent, -maximum_speed*tangent), dim=2)
    choices = torch.tensor(list(product(range(5), repeat=3)), device=p.device)
    commands = local[:, torch.arange(3, device=p.device)[None, :], choices]
    path = constant_command_path(p, v, commands, dt=dt, steps=steps, kp=velocity_kp, acceleration=acceleration)
    target_min, violation = path_violations(path, target, tv, obs, radii, dt=dt,
        target_separation=prediction_separation, agent_separation=agent_separation,
        pursuer_radius=pursuer_radius, obstacle_clearance=obstacle_clearance,
        arena_half_extent=arena_half_extent, boundary_clearance=boundary_clearance,
        target_acceleration=target_acceleration[ids] if target_acceleration is not None else None,
        target_prediction_speed=target_prediction_speed)
    admissible = (violation <= violation[:, :1]+1e-6).all(dim=-1)
    target_count = target_min.shape[-1]
    score = violation[..., :target_count].sum(dim=-1)
    scored = torch.where(admissible, score, torch.inf)
    best = scored.amin(dim=1, keepdim=True)
    change = (commands-nominal[:, None]).square().sum(dim=(-1, -2))
    eligible = admissible & (scored <= best+1e-7)
    if margin_tiebreak:
        margin = target_min.amin(dim=-1)
        best_margin = torch.where(eligible, margin, -torch.inf).amax(dim=1, keepdim=True)
        eligible &= margin >= best_margin-1e-6
    choice = torch.where(eligible, change, torch.inf).argmin(dim=1)
    selected_score = score.gather(1, choice[:, None]).squeeze(1)
    improve = ((violation[:, 0, :target_count].amax(dim=1) > 1e-4)
               & (score[:, 0]-selected_score > 1e-6))
    choice = torch.where(improve, choice, torch.zeros_like(choice))
    selected = commands[torch.arange(len(ids), device=p.device), choice]
    selected_min = target_min[torch.arange(len(ids), device=p.device), choice].amin(dim=-1)
    result.velocity_xy[ids] = selected
    result.corrected[ids] = improve
    result.selected_index[ids] = choice
    result.nominal_minimum_m[ids] = target_min[:, 0].amin(dim=-1)
    result.selected_minimum_m[ids] = selected_min
    result.selected_infeasible[ids] = selected_min < prediction_separation-1e-4
    result.maximum_agent_correction_mps[ids] = (selected-nominal).norm(dim=-1).amax(dim=-1)
    return result
