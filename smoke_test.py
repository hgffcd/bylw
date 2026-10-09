# -*- coding: utf-8 -*-
"""冒烟测试: 验证从环境到训练管线各环节可运行.

云端/无torch环境:  python smoke_test.py --env-only
本机 rl_env 完整:  python smoke_test.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def env_smoke():
    from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv
    t0 = time.time()
    env = PursuitLostEnv(EnvConfig(seed=0, n_dynamic_obs=2))
    obs, state = env.reset()
    steps, ep = 0, 0
    for _ in range(2000):
        obs, state, rew, done, info = env.step(np.random.uniform(-1, 1, (env.n, 2)))
        steps += 1
        if done:
            ep += 1
            env.reset()
    sps = steps / (time.time() - t0)
    print(f"[env] obs_dim={env.obs_dim} state_dim={env.state_dim} "
          f"steps/s={sps:.0f} episodes={ep}  OK")
    from safety.cbf_qp import CBFSafetyLayer
    layer = CBFSafetyLayer()
    u, c = layer.project(np.array([1.0, 0.0]), np.zeros(2), 0.0, 4.0, [],
                         [(np.array([2.5, 0.0]), 1.0, np.zeros(2))])
    print(f"[safety] corrected={c} u={u.round(3)}  OK")


def train_smoke():
    import torch
    from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv
    from algos.mappo import MAPPO, RolloutBuffer
    env_cfg = EnvConfig(seed=0, obs_mode="oracle", sense_range=200.0, fov_half=3.15)
    n_env = 2
    envs = [PursuitLostEnv(EnvConfig(**{**env_cfg.__dict__, "seed": k})) for k in range(n_env)]
    probe = envs[0]
    agent = MAPPO(probe.obs_dim, probe.state_dim, probe.act_dim, probe.n,
                  hidden=64, epochs=2, minibatch=2, device="cpu")
    obs = np.stack([e.reset()[0] for e in envs])
    state = np.stack([e._get_state() for e in envs])
    for it in range(3):
        buf = RolloutBuffer(32, n_env, probe.n, probe.obs_dim, probe.state_dim,
                            probe.act_dim, agent.device)
        for _ in range(32):
            o = torch.as_tensor(obs, dtype=torch.float32)
            s = torch.as_tensor(state, dtype=torch.float32)
            act, logp, val = agent.act(o, s)
            rew = np.zeros(n_env, np.float32)
            dn = np.zeros(n_env, np.float32)
            a = act.numpy()
            for k, env in enumerate(envs):
                ob, st, r, done, _ = env.step(a[k])
                rew[k] = r[0]
                if done:
                    ob, st = env.reset()
                    dn[k] = 1
                obs[k], state[k] = ob, st
            buf.add(o, s, act, logp, torch.as_tensor(rew), torch.as_tensor(dn), val)
        with torch.no_grad():
            last = agent.critic(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)
        info = agent.update(buf, last)
        assert all(np.isfinite(v) for v in info.values()), info
        print(f"[mappo] iter{it}: {info}  OK")

    from belief.gru_predictor import GRUGaussianPredictor
    m = GRUGaussianPredictor()
    seq = torch.randn(4, 20, 5)
    mu, var, _ = m(seq)
    loss = m.nll_loss(mu, var, torch.randn(4, 5, 2))
    loss.backward()
    mus, vars_ = m.rollout_lost(seq, steps=8)
    assert torch.isfinite(loss) and torch.isfinite(mus).all()
    print(f"[belief] nll={float(loss):.3f} rollout_var_growth="
          f"{float(vars_[0,0].mean()):.3f}->{float(vars_[0,-1].mean()):.3f}  OK")

    from algos.networks import ConfidenceGAT
    gat = ConfidenceGAT(feat_dim=8)
    nodes = torch.randn(4, 6, 8)
    out_hi = gat(nodes, conf=torch.ones(4))
    out_lo = gat(nodes, conf=torch.full((4,), 1e-3))
    assert out_hi.shape == (4, 64) and not torch.allclose(out_hi, out_lo)
    print("[gat] confidence modulation changes output  OK")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-only", action="store_true")
    args = ap.parse_args()
    env_smoke()
    if not args.env_only:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("torch 不可用, 跳过训练冒烟(请在 rl_env 中运行完整版)")
            sys.exit(0)
        train_smoke()
