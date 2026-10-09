# -*- coding: utf-8 -*-
"""MAPPO 基线 (参数共享 actor + 集中式 critic), 对应开题报告 5.3.1 节.

- actor: MLP + GRU 可选(默认MLP), 输出 tanh 压缩高斯策略, 动作 [-1,1]^2;
- critic: 以全局状态为输入的 MLP;
- GAE + PPO clip 更新, 优势小批量标准化;
- 网络结构处预留 GAT/置信调制接口 (encoder 可替换, 见 networks.py).

依赖 torch —— 请在本机 rl_env 中运行; 云端仅做语法校验.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from algos.networks import CentralCritic, build_gaussian_actor
from algos.ppo_diagnostics import active_policy_sums, explained_variance, grad_norm


def expand_mlp_actor_observation_state(
    actor_state: dict[str, torch.Tensor],
    new_observation_dim: int,
) -> dict[str, torch.Tensor]:
    """Zero-pad an MLP actor input while preserving its exact old policy."""
    key = "body.0.weight"
    if key not in actor_state:
        raise ValueError("actor checkpoint has no MLP input layer")
    old_weight = actor_state[key]
    if old_weight.ndim != 2:
        raise ValueError("actor MLP input weight must be a matrix")
    old_observation_dim = old_weight.shape[1]
    if new_observation_dim < old_observation_dim:
        raise ValueError("new observation dimension may not shrink")
    expanded = {
        name: tensor.detach().clone() for name, tensor in actor_state.items()
    }
    if new_observation_dim == old_observation_dim:
        return expanded
    padded = old_weight.new_zeros(old_weight.shape[0], new_observation_dim)
    padded[:, :old_observation_dim] = old_weight
    expanded[key] = padded
    return expanded


class RolloutBuffer:
    def __init__(self, T, n_env, n_agent, obs_dim, state_dim, act_dim, device):
        self.T, self.n_env, self.n_agent = T, n_env, n_agent
        self.obs = torch.zeros(T, n_env, n_agent, obs_dim, device=device)
        self.state = torch.zeros(T, n_env, state_dim, device=device)
        self.act = torch.zeros(T, n_env, n_agent, act_dim, device=device)
        self.raw_act = torch.zeros_like(self.act)
        self.has_raw_actions = False
        self.logp = torch.zeros(T, n_env, n_agent, device=device)
        self.rew = torch.zeros(T, n_env, device=device)
        self.done = torch.zeros(T, n_env, device=device)
        # Optional environment-level actor mask for hierarchical controllers.
        # One value applies to all parameter-sharing agents in an environment:
        # 0 means a traditional supervisor executed the command, so that sample
        # may train the critic but must not update the actor.
        self.policy_mask = torch.ones(T, n_env, device=device)
        self.val = torch.zeros(T + 1, n_env, device=device)
        self.ptr = 0

    def add(self, obs, state, act, logp, rew, done, val, policy_mask=None,
            raw_action=None):
        t = self.ptr
        if t and self.has_raw_actions != (raw_action is not None):
            raise ValueError("raw actions must be supplied for the entire rollout")
        self.has_raw_actions = raw_action is not None
        if raw_action is not None:
            if raw_action.shape != self.act[t].shape:
                raise ValueError("raw_action shape mismatch")
            self.raw_act[t] = raw_action
        self.obs[t] = obs
        self.state[t] = state
        self.act[t] = act
        self.logp[t] = logp
        self.rew[t] = rew
        self.done[t] = done
        if policy_mask is not None:
            if policy_mask.shape != (self.n_env,):
                raise ValueError("policy_mask must have shape [n_env]")
            self.policy_mask[t] = policy_mask.to(
                device=self.policy_mask.device, dtype=self.policy_mask.dtype
            )
        self.val[t] = val
        self.ptr += 1

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        self.val[self.T] = last_val
        adv = torch.zeros_like(self.rew)
        gae = torch.zeros(self.n_env, device=self.rew.device)
        for t in reversed(range(self.T)):
            mask = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * self.val[t + 1] * mask - self.val[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
        ret = adv + self.val[: self.T]
        return adv, ret


class MAPPO:
    def __init__(self, obs_dim, state_dim, act_dim, n_agent,
                 hidden=128, lr=3e-4, clip=0.2, epochs=10, minibatch=4,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, device="cpu",
                 actor_lr=None, critic_lr=None, actor_encoder="mlp",
                 actor_encoder_kwargs=None, policy_anchor_coef=0.0,
                 collect_diagnostics=False):
        if policy_anchor_coef < 0:
            raise ValueError("policy_anchor_coef must be non-negative")
        self.device = torch.device(device)
        self.actor = build_gaussian_actor(
            obs_dim,
            act_dim,
            hidden=hidden,
            n_agent=n_agent,
            actor_encoder=actor_encoder,
            actor_encoder_kwargs=actor_encoder_kwargs,
        ).to(self.device)
        self.critic = CentralCritic(state_dim, hidden * 2).to(self.device)
        # actor/critic 学习率可分离(BC热启动微调需 actor 慢 critic 快)
        self.opt = torch.optim.Adam([
            {"params": self.actor.parameters(), "lr": actor_lr or lr},
            {"params": self.critic.parameters(), "lr": critic_lr or lr},
        ])
        self.clip, self.epochs, self.minibatch = clip, epochs, minibatch
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_agent = n_agent
        self.policy_anchor_coef = float(policy_anchor_coef)
        self.policy_anchor_actor = None
        self.collect_diagnostics = bool(collect_diagnostics)

    def set_policy_anchor(self):
        """Freeze the current actor as the behavior prior for later PPO updates."""
        self.policy_anchor_actor = copy.deepcopy(self.actor).to(self.device)
        self.policy_anchor_actor.eval()
        self.policy_anchor_actor.requires_grad_(False)

    def policy_anchor_kl(self, obs, mask=None):
        """KL(anchor || current) on pre-tanh Gaussians.

        Both policies share the same bijective tanh transform, so the
        pre-transform KL is also the transformed-policy KL away from numerical
        saturation and avoids sampling noise in the anchor loss.
        """
        if self.policy_anchor_actor is None:
            return obs.new_zeros(())
        mean, log_std = self.actor(obs)
        with torch.no_grad():
            anchor_mean, anchor_log_std = self.policy_anchor_actor(obs)
        divergence = kl_divergence(
            Normal(anchor_mean, anchor_log_std.exp()),
            Normal(mean, log_std.exp()),
        ).sum(-1)
        if mask is None:
            return divergence.mean()
        if mask.shape != divergence.shape:
            raise ValueError("policy anchor mask shape mismatch")
        return (divergence * mask).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def act(self, obs, state, return_raw=False):
        """obs: (n_env, n_agent, obs_dim); state: (n_env, state_dim)."""
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw = dist.rsample()
        act = torch.tanh(raw)
        logp = (
            self._squashed_log_probability(dist, raw)
            if return_raw else
            (dist.log_prob(raw) - torch.log1p(-act.pow(2) + 1e-6)).sum(-1)
        )
        val = self.critic(state).squeeze(-1)
        if return_raw:
            return act, logp, val, raw
        return act, logp, val

    @staticmethod
    def _squashed_log_probability(dist, raw):
        # log(1 - tanh(x)^2), stable even when float32 tanh rounds to +/-1.
        log_jacobian = 2.0 * (np.log(2.0) - raw - F.softplus(-2.0 * raw))
        return (dist.log_prob(raw) - log_jacobian).sum(-1)

    def evaluate(self, obs, act, raw_action=None):
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        if raw_action is None:
            raw = torch.atanh(act.clamp(-0.999, 0.999))
            logp = (dist.log_prob(raw) - torch.log1p(-act.pow(2) + 1e-6)).sum(-1)
        else:
            logp = self._squashed_log_probability(dist, raw_action)
        ent = dist.entropy().sum(-1)
        return logp, ent

    def update(self, buf: RolloutBuffer, last_val, gamma=0.99, lam=0.95):
        adv, ret = buf.compute_gae(last_val, gamma, lam)
        T, E, A = buf.T, buf.n_env, self.n_agent
        obs = buf.obs.reshape(T * E, A, -1)
        state = buf.state.reshape(T * E, -1)
        act = buf.act.reshape(T * E, A, -1)
        raw_act = buf.raw_act.reshape(T * E, A, -1) if buf.has_raw_actions else None
        old_logp = buf.logp.reshape(T * E, A)
        adv_f = adv.reshape(T * E, 1).expand(-1, A)
        ret_f = ret.reshape(T * E)
        policy_mask = buf.policy_mask.reshape(T * E, 1).expand(-1, A)
        active_advantage = adv_f[policy_mask.bool()]
        if active_advantage.numel():
            active_mean = active_advantage.mean()
            active_std = active_advantage.std(unbiased=False)
            adv_f = (adv_f - active_mean) / (active_std + 1e-8)
        else:
            adv_f = torch.zeros_like(adv_f)
        idx = np.arange(T * E)
        if self.policy_anchor_coef > 0 and self.policy_anchor_actor is None:
            raise RuntimeError(
                "policy anchor coefficient is positive but no anchor was set")
        info = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, anchor_kl=0.0)
        if self.collect_diagnostics:
            diagnostics=dict(kl_sum=0.,clip_sum=0.,active_count=0,actor_norm=0.,critic_norm=0.,clip_scale=0.,clipped_batches=0)
            with torch.no_grad():
                explained_before=explained_variance(ret_f,self.critic(state).squeeze(-1))
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for chunk in np.array_split(idx, self.minibatch):
                b = torch.as_tensor(chunk, device=self.device)
                logp, ent = self.evaluate(
                    obs[b], act[b], raw_action=raw_act[b] if raw_act is not None else None
                )
                ratio = (logp - old_logp[b]).exp()                       # 式(5-13)
                s1 = ratio * adv_f[b]
                s2 = ratio.clamp(1 - self.clip, 1 + self.clip) * adv_f[b]
                batch_policy_mask = policy_mask[b]
                active_count = batch_policy_mask.sum()
                if self.collect_diagnostics:
                    kl_sum,clip_sum,count=active_policy_sums(logp-old_logp[b],batch_policy_mask,self.clip)
                    diagnostics['kl_sum']+=kl_sum;diagnostics['clip_sum']+=clip_sum;diagnostics['active_count']+=count
                if active_count.item() > 0:
                    pi_loss = -(
                        torch.min(s1, s2) * batch_policy_mask
                    ).sum() / active_count                                # 式(5-14)
                    entropy = (ent * batch_policy_mask).sum() / active_count
                else:
                    # Keep a differentiable zero while preventing traditional
                    # search/approach samples from changing actor parameters.
                    pi_loss = logp.sum() * 0.0
                    entropy = ent.sum() * 0.0
                v = self.critic(state[b]).squeeze(-1)
                v_loss = (v - ret_f[b]).pow(2).mean()                     # 价值损失
                anchor_kl = (
                    self.policy_anchor_kl(obs[b], batch_policy_mask)
                    if active_count.item() > 0
                    else obs[b].sum() * 0.0
                )
                loss = (
                    pi_loss + self.vf_coef * v_loss
                    - self.ent_coef * entropy
                    + self.policy_anchor_coef * anchor_kl
                )
                self.opt.zero_grad()
                loss.backward()
                if active_count.item() == 0:
                    # Adam updates parameters with a zero gradient when it has
                    # non-zero momentum from an earlier active minibatch.  A
                    # None gradient is required to make a traditional-only
                    # minibatch a strict no-op for the actor.
                    for parameter in self.actor.parameters():
                        parameter.grad = None
                if self.collect_diagnostics:
                    diagnostics['actor_norm']+=grad_norm(self.actor.parameters())
                    diagnostics['critic_norm']+=grad_norm(self.critic.parameters())
                total_grad_norm = nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                if self.collect_diagnostics:
                    scale=min(1.,self.max_grad_norm/(float(total_grad_norm)+1e-6))
                    diagnostics['clip_scale']+=scale
                    diagnostics['clipped_batches']+=int(scale<1.)
                self.opt.step()
                info["pi_loss"] += float(pi_loss.detach())
                info["v_loss"] += float(v_loss.detach())
                info["entropy"] += float(entropy.detach())
                info["anchor_kl"] += float(anchor_kl.detach())
        k = self.epochs * self.minibatch
        result = {m: v / k for m, v in info.items()}
        result["policy_active_fraction"] = float(buf.policy_mask.mean().item())
        if self.collect_diagnostics:
            count=diagnostics['active_count']
            with torch.no_grad():
                explained_after=explained_variance(ret_f,self.critic(state).squeeze(-1))
            result['diagnostics']=dict(
                approx_kl_active=diagnostics['kl_sum']/count if count else None,
                clip_fraction_active=diagnostics['clip_sum']/count if count else None,
                repeated_active_agent_samples=count,
                actor_grad_norm_mean=diagnostics['actor_norm']/k,
                critic_grad_norm_mean=diagnostics['critic_norm']/k,
                global_clip_scale_mean=diagnostics['clip_scale']/k,
                grad_clipped_batch_fraction=diagnostics['clipped_batches']/k,
                critic_explained_variance_before=explained_before,
                critic_explained_variance_after=explained_after)
        return result

    def save(self, path):
        torch.save(dict(actor=self.actor.state_dict(),
                        critic=self.critic.state_dict()), path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
