"""Read-only PPO diagnostics. None denotes an undefined statistic."""
import torch


def grad_norm(parameters):
    norms=[p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    return float(torch.stack(norms).norm(2)) if norms else 0.


def explained_variance(target,prediction):
    variance=target.detach().var(unbiased=False)
    if variance <= 1e-12:return None
    return float(1-(target.detach()-prediction.detach()).var(unbiased=False)/variance)


def active_policy_sums(log_ratio,mask,clip):
    mask=mask.to(dtype=torch.bool)
    value=log_ratio.detach()[mask]
    if not value.numel():return 0.,0.,0
    kl=(torch.expm1(value)-value).sum()
    clipped=((value.exp()-1).abs()>clip).sum()
    return float(kl),float(clipped),value.numel()
