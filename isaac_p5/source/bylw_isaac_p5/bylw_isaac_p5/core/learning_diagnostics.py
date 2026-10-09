"""Same-state slot-controller action audit; no second live environment call."""
import torch
from .native_training import moving_target_slot_guidance


def measured_slot_guidance(*args,**kwargs):
    # Snapshot BEFORE the actual call; the shadow call owns all its tensors.
    clone=lambda x:x.detach().clone() if isinstance(x,torch.Tensor) else x
    shadow_args=tuple(clone(x) for x in args)
    shadow_kwargs={k:clone(v) for k,v in kwargs.items()}
    for key in ('gain_residual_action','radial_velocity_residual_action','tangential_velocity_residual_action'):
        if shadow_kwargs.get(key) is not None:shadow_kwargs[key]=torch.zeros_like(shadow_kwargs[key])
    actual_sink,shadow_sink={},{}
    result=moving_target_slot_guidance(*args,**kwargs,diagnostic_sink=actual_sink)
    shadow=moving_target_slot_guidance(*shadow_args,**shadow_kwargs,diagnostic_sink=shadow_sink)
    scale=torch.as_tensor(kwargs['policy_residual_scale'],device=result.active.device)
    active=result.active & (scale>0)
    mask=active[:,None,None]
    result.learning_diagnostics=dict(
        learning_diag_slot_active=active.detach().clone(),
        learning_diag_slot_velocity=result.velocity_xy.detach().clone(),
        learning_diag_slot_zero_action_velocity=shadow.velocity_xy.detach().clone(),
        learning_diag_slot_effect_before_speed=torch.where(mask,actual_sink['unlimited_velocity_xy']-shadow_sink['unlimited_velocity_xy'],0.),
        learning_diag_slot_effect_after_speed=torch.where(mask,result.velocity_xy.detach()-shadow.velocity_xy.detach(),0.))
    return result
