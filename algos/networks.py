# -*- coding: utf-8 -*-
"""网络结构: 高斯actor / 集中critic / 置信度调制GAT(对应开题5.3.2节).

GAT接口说明: ConfidenceGAT 以实体节点特征为输入, 目标信念节点的注意力
logit 加上 log(conf) 偏置(式5-17). 基线阶段 actor 用 MLP 编码原始观测;
PursuitGraphEncoder 将冻结的扁平观测解析成实体节点，并保持不确定区域节点不被
低置信度整体丢弃。默认 actor_encoder="mlp" 与历史checkpoint完全兼容.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, encoder: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.out_dim if encoder is not None else obs_dim
        self.body = mlp([in_dim, hidden, hidden])
        self.act_body = nn.Tanh()
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs):
        x = self.encoder(obs) if self.encoder is not None else obs
        h = self.act_body(self.body(x))
        mu = self.mu(h)
        # 钳制防止熵奖励无界推高 log_std (M1 v1/v2a 观察到熵失控上行)
        log_std = self.log_std.clamp(-3.0, 0.2)
        return mu, log_std.expand_as(mu)


class CentralCritic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.v = mlp([state_dim, hidden, hidden, 1])

    def forward(self, state):
        return self.v(state)


class ConfidenceGAT(nn.Module):
    """置信度调制的图注意力编码器(轻量实现, 第4章创新点2的网络载体).

    输入约定: nodes (..., n_node, feat) 为实体节点特征, 其中
    node 0 = 自身, node 1 = 目标信念节点, 其余为队友/障碍;
    conf (...,) 为目标信念置信度 c_t, 以对数偏置参与注意力(式5-17).
    """

    def __init__(self, feat_dim, out_dim=64, n_layers=2):
        super().__init__()
        self.out_dim = out_dim
        self.W = nn.ModuleList([nn.Linear(feat_dim if i == 0 else out_dim, out_dim)
                                for i in range(n_layers)])
        self.a = nn.ParameterList([nn.Parameter(torch.empty(2 * out_dim))
                                   for _ in range(n_layers)])
        for p in self.a:
            nn.init.normal_(p, std=0.1)
        self.last_attention = None

    def forward(self, nodes, conf=None):
        h = nodes
        for W, a in zip(self.W, self.a):
            h = W(h)                                                    # (..., n, d)
            q = h[..., :1, :]                                           # 自身节点为 query
            e = F.leaky_relu(
                (torch.cat([q.expand_as(h), h], dim=-1) * a).sum(-1), 0.2)   # 式(5-16)
            if conf is not None:
                bias = torch.zeros_like(e)
                bias[..., 1] = torch.log(conf.clamp_min(1e-6))          # 信念节点对数偏置(式5-17)
                e = e + bias
            alpha = torch.softmax(e, dim=-1).unsqueeze(-1)
            self.last_attention = alpha.squeeze(-1).detach()
            agg = (alpha * h).sum(-2, keepdim=True)                     # 式(5-18)
            h = torch.tanh(h + agg)                                     # 残差 + 广播
        return h[..., 0, :]                                             # 自身节点表征


class PursuitGraphEncoder(nn.Module):
    """Parse the flat pursuit observation into typed graph entities.

    Node order is fixed and auditable:
      0 self, 1 target mean, 2 target uncertainty region,
      3.. teammates, then nearest obstacles.

    Confidence is *not* copied into node features. Standard GAT therefore sees
    the same nodes as ConfidenceGAT; the only changed mechanism is the
    ``log(conf)`` bias on the target-mean node. Covariance summary and
    ``t_lost`` remain in the separate uncertainty node.
    """

    MODES = {"deep_sets", "gat", "confidence_gat"}

    def __init__(
        self,
        obs_dim: int,
        n_agent: int,
        out_dim: int = 64,
        n_layers: int = 2,
        mode: str = "confidence_gat",
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f"graph encoder mode must be one of {sorted(self.MODES)}, got {mode!r}")
        base = 5 + 6 + 4 * (n_agent - 1)
        remaining = obs_dim - base
        if remaining < 0 or remaining % 5:
            raise ValueError(
                f"obs_dim={obs_dim} is incompatible with n_agent={n_agent}")
        self.obs_dim = int(obs_dim)
        self.n_agent = int(n_agent)
        self.n_teammates = n_agent - 1
        self.n_obstacles = remaining // 5
        self.n_nodes = 3 + self.n_teammates + self.n_obstacles
        self.mode = mode
        self.out_dim = int(out_dim)
        self.type_dim = 5
        self.payload_dim = 7
        self.feat_dim = self.type_dim + self.payload_dim
        if mode == "deep_sets":
            self.node_embed = mlp([self.feat_dim, out_dim, out_dim], nn.ReLU)
            self.combine = mlp([2 * out_dim, out_dim, out_dim], nn.ReLU)
            self.gat = None
        else:
            self.node_embed = None
            self.combine = None
            self.gat = ConfidenceGAT(
                self.feat_dim, out_dim=out_dim, n_layers=n_layers)

    @staticmethod
    def _put_node(nodes, index, type_index, payload):
        nodes[..., index, type_index] = 1.0
        nodes[..., index, 5: 5 + payload.shape[-1]] = payload

    def build_nodes(self, obs):
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"expected obs last dim {self.obs_dim}, got {obs.shape[-1]}")
        nodes = obs.new_zeros(*obs.shape[:-1], self.n_nodes, self.feat_dim)

        self._put_node(nodes, 0, 0, obs[..., :5])

        target = obs[..., 5:11]
        visible = target[..., 4] > 0
        zeros = torch.zeros_like(target[..., 2])
        confidence = torch.where(
            visible, torch.ones_like(target[..., 2]), target[..., 2]
        ).clamp(0.0, 1.0)
        vel_x = torch.where(visible, target[..., 2], zeros)
        vel_y = torch.where(visible, target[..., 3], zeros)
        sigma = torch.where(visible, zeros, target[..., 3].abs())
        mean_payload = torch.stack(
            [
                target[..., 0],
                target[..., 1],
                vel_x,
                vel_y,
                target[..., 4],
                target[..., 5],
                zeros,
            ],
            dim=-1,
        )
        uncertainty_payload = torch.stack(
            [
                sigma,
                target[..., 5],
                (~visible).to(target.dtype),
                zeros,
                zeros,
                zeros,
                zeros,
            ],
            dim=-1,
        )
        self._put_node(nodes, 1, 1, mean_payload)
        self._put_node(nodes, 2, 2, uncertainty_payload)

        offset = 11
        node_index = 3
        for _ in range(self.n_teammates):
            teammate = obs[..., offset: offset + 4]
            self._put_node(nodes, node_index, 3, teammate)
            offset += 4
            node_index += 1
        for _ in range(self.n_obstacles):
            obstacle = obs[..., offset: offset + 5]
            self._put_node(nodes, node_index, 4, obstacle)
            offset += 5
            node_index += 1
        return nodes, confidence

    def forward(self, obs):
        nodes, confidence = self.build_nodes(obs)
        if self.mode == "deep_sets":
            embedded = self.node_embed(nodes)
            pooled = embedded.mean(dim=-2)
            return self.combine(torch.cat([embedded[..., 0, :], pooled], dim=-1))
        conf = confidence if self.mode == "confidence_gat" else None
        return self.gat(nodes, conf=conf)


def build_gaussian_actor(
    obs_dim: int,
    act_dim: int,
    hidden: int = 128,
    n_agent: int = 3,
    actor_encoder: str = "mlp",
    actor_encoder_kwargs: dict | None = None,
) -> GaussianActor:
    """Single construction path shared by training, evaluation and rendering."""
    kwargs = dict(actor_encoder_kwargs or {})
    if actor_encoder == "mlp":
        if kwargs:
            raise ValueError("actor_encoder_kwargs require a non-MLP encoder")
        encoder = None
    else:
        encoder = PursuitGraphEncoder(
            obs_dim=obs_dim,
            n_agent=n_agent,
            mode=actor_encoder,
            **kwargs,
        )
    return GaussianActor(obs_dim, act_dim, hidden, encoder=encoder)
