"""C101 bounded joint command search with an acceleration-limited predictor.

This is a local model-based correction, not a trapping/visibility guarantee.
The nominal controller and final safety projections remain in the control chain.
"""
from dataclasses import dataclass
from itertools import product
import math
import torch


@dataclass
class JointLookaheadResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    maximum_agent_correction_mps: torch.Tensor
    nominal_violation: torch.Tensor
    selected_violation: torch.Tensor
    selected_index: torch.Tensor


def predict_joint_positions(position, velocity, command, *, dt, steps, kp, acceleration, sample_every):
    """Roll candidate constant commands through the existing planar velocity model."""
    p=position[:, None].expand_as(command).clone()
    v=velocity[:, None].expand_as(command).clone()
    samples=[]
    for step in range(steps):
        a=kp*(command-v)
        a=a*(acceleration/a.norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp_max(1)
        v=v+dt*a
        p=p+dt*v
        if (step+1)%sample_every==0:samples.append(p)
    return torch.stack(samples,dim=2)


def joint_region_lookahead(nominal_velocity, current_velocity, pursuer_xy,
                          target_xy, target_velocity, active, obstacle_xy,
                          obstacle_radius, *, maximum_speed=4., correction_bound=.75,
                          horizon=.5, low_level_dt=.01, velocity_kp=2., acceleration=4.,
                          sample_dt=.1, capture_radius=5.5, maximum_gap=5*math.pi/6,
                          target_separation=1.5, agent_separation=3.5,
                          pursuer_radius=.5, obstacle_clearance=1.5,
                          arena_half_extent=50., boundary_clearance=1.):
    """Choose among 5^3 nearby joint commands using current shared measurements.

    Each agent keeps its command or adds +/-0.75 m/s along the measured radial
    or tangential direction, then projects onto the unchanged speed ball. The
    original command is candidate zero. A candidate may not worsen either
    terminal task violation or any sampled safety/LOS violation relative to
    nominal. The smallest task violation wins; ties retain the least change.
    Target prediction is bounded-horizon CV, with no future opponent state.
    """
    if pursuer_xy.ndim!=3 or pursuer_xy.shape[1:]!=(3,2):
        raise ValueError('joint lookahead requires three XY agents')
    b=pursuer_xy.shape[0]
    if (nominal_velocity.shape!=pursuer_xy.shape or current_velocity.shape!=pursuer_xy.shape
            or target_xy.shape!=(b,2) or target_velocity.shape!=(b,2)
            or active.shape!=(b,) or active.dtype!=torch.bool):
        raise ValueError('joint lookahead state shape or mask mismatch')
    if (min(maximum_speed,correction_bound,horizon,low_level_dt,velocity_kp,
            acceleration,sample_dt,target_separation,agent_separation,pursuer_radius)>0
            and capture_radius>target_separation and 2*math.pi/3<maximum_gap<math.pi):
        pass
    else:raise ValueError('invalid joint lookahead scales')
    steps=round(horizon/low_level_dt);stride=round(sample_dt/low_level_dt)
    if (stride<1 or steps%stride or not math.isclose(steps*low_level_dt,horizon,abs_tol=1e-8)
            or not math.isclose(stride*low_level_dt,sample_dt,abs_tol=1e-8)):
        raise ValueError('lookahead sample times must divide the physical step and horizon')
    if obstacle_xy.ndim!=3 or obstacle_xy.shape[0]!=b or obstacle_xy.shape[-1]!=2:
        raise ValueError('obstacle shape mismatch')
    if obstacle_radius.shape==(obstacle_xy.shape[1],):obstacle_radius=obstacle_radius[None].expand(b,-1)
    if obstacle_radius.shape!=obstacle_xy.shape[:2]:raise ValueError('obstacle radius shape mismatch')
    finite=(torch.isfinite(pursuer_xy).all((1,2)) & torch.isfinite(current_velocity).all((1,2))
            & torch.isfinite(nominal_velocity).all((1,2)) & torch.isfinite(target_xy).all(1)
            & torch.isfinite(target_velocity).all(1) & torch.isfinite(obstacle_xy).all((1,2))
            & torch.isfinite(obstacle_radius).all(1))
    valid=active & finite
    zeros=nominal_velocity.new_zeros(b)
    result=JointLookaheadResult(nominal_velocity.clone(),valid,torch.zeros_like(valid),
                               zeros.clone(),zeros.clone(),zeros.clone(),torch.zeros_like(active,dtype=torch.long))
    if not valid.any():return result
    ids=valid.nonzero().squeeze(1)
    p=pursuer_xy[ids];v=current_velocity[ids];nominal=nominal_velocity[ids]
    target=target_xy[ids];tv=target_velocity[ids];obs=obstacle_xy[ids];radii=obstacle_radius[ids]
    radial=p-target[:,None];radial=radial/radial.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    tangent=torch.stack((-radial[...,1],radial[...,0]),dim=-1)
    local=torch.stack((torch.zeros_like(radial),radial,-radial,tangent,-tangent),dim=2)
    choices=torch.tensor(list(product(range(5),repeat=3)),device=p.device)
    directions=local[:,torch.arange(3,device=p.device)[None,:],choices]
    commands=nominal[:,None]+correction_bound*directions
    commands=commands*(maximum_speed/commands.norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp_max(1)
    commands[:,0]=nominal  # Preserve the baseline exactly, including rounding.
    trajectory=predict_joint_positions(p,v,commands,dt=low_level_dt,steps=steps,
                                       kp=velocity_kp,acceleration=acceleration,sample_every=stride)
    times=torch.arange(1,steps//stride+1,device=p.device,dtype=p.dtype)*sample_dt
    targets=target[:,None]+times[None,:,None]*tv[:,None]
    relative=trajectory-targets[:,None,:,None]
    distance=relative.norm(dim=-1)
    terminal=relative[:,:,-1]
    angles=torch.atan2(terminal[...,1],terminal[...,0]).sort(dim=-1).values
    gaps=torch.cat((angles[...,1:]-angles[...,:-1],2*math.pi-angles[...,-1:]+angles[...,:1]),dim=-1)
    radius_violation=((distance[:,:,-1].amax(-1)-capture_radius)/capture_radius).clamp_min(0)
    gap_violation=((gaps.amax(-1)-maximum_gap)/(2*math.pi-maximum_gap)).clamp_min(0)
    task=torch.stack((radius_violation,gap_violation),dim=-1)
    target_bad=(target_separation-distance).clamp_min(0).amax((-1,-2))
    pairs=trajectory[...,torch.tensor([0,0,1],device=p.device),:]-trajectory[...,torch.tensor([1,2,2],device=p.device),:]
    pair_bad=(agent_separation-pairs.norm(dim=-1)).clamp_min(0).amax((-1,-2))
    boundary_bad=(trajectory.abs()-(arena_half_extent-boundary_clearance)).clamp_min(0).amax((-1,-2,-3))
    if obs.shape[1]:
        delta=trajectory[:,:,:,:,None]-obs[:,None,None,None]
        obstacle_bad=(pursuer_radius+obstacle_clearance+radii[:,None,None,None]-delta.norm(dim=-1)).clamp_min(0).amax((-1,-2,-3))
        segment=targets[:,None,:,None]-trajectory
        fraction=(-(delta*segment[...,None,:]).sum(-1)/segment.square().sum(-1)[...,None].clamp_min(1e-9)).clamp(0,1)
        closest=delta+fraction[...,None]*segment[...,None,:]
        los_bad=(radii[:,None,None,None]-closest.norm(dim=-1)).clamp_min(0).amax((-1,-2,-3))
    else:obstacle_bad=torch.zeros_like(target_bad);los_bad=torch.zeros_like(target_bad)
    safety=torch.stack((target_bad,pair_bad,obstacle_bad,boundary_bad,los_bad),dim=-1)
    admissible=((task<=task[:,:1]+1e-6).all(-1) & (safety<=safety[:,:1]+1e-6).all(-1))
    score=task.sum(-1)
    # Monotone task improvement first, minimal intervention among numerical ties.
    scored=torch.where(admissible,score,torch.inf)
    best=scored.amin(-1,keepdim=True)
    change=(commands-nominal[:,None]).square().sum((-1,-2))
    choice=torch.where(admissible & (scored<=best+1e-7),change,torch.inf).argmin(-1)
    improve=score[:,0]-score.gather(1,choice[:,None]).squeeze(1)>1e-6
    choice=torch.where(improve,choice,torch.zeros_like(choice))
    selected=commands[torch.arange(len(ids),device=p.device),choice]
    result.velocity_xy[ids]=selected
    result.corrected[ids]=improve
    result.maximum_agent_correction_mps[ids]=(selected-nominal).norm(dim=-1).amax(-1)
    result.nominal_violation[ids]=score[:,0]
    result.selected_violation[ids]=score.gather(1,choice[:,None]).squeeze(1)
    result.selected_index[ids]=choice
    return result
