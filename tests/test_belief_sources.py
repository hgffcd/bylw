# -*- coding: utf-8 -*-
"""论文A S-A1/S-A2 信念来源模块单测(纯numpy): pytest tests -q

覆盖: belief/imm_filter.py (IMM CV+CT) / belief/kf_single.py (单模型CV-KF)
     / belief/cv_extrap.py (匀速外推). 不触碰 tests/test_env.py 的环境测试.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belief.cv_extrap import CVExtrapolator
from belief.imm_filter import IMMFilter
from belief.kf_single import KFSingleCV

DT = 0.1


def make_cv_ct_traj(n_cv=100, n_ct=150, omega=0.3, speed=5.0, dt=DT, heading0=0.3):
    """合成 CV->CT 切换轨迹: 前 n_cv 步匀速直线, 后 n_ct 步恒转弯率盘旋.

    返回 (pos (N,2), vel (N,2)); 切换发生在下标 n_cv 处.
    """
    pos = np.array([20.0, 20.0])
    heading = heading0
    P, V = [], []
    for k in range(n_cv + n_ct):
        if k >= n_cv:
            heading += omega * dt
        vel = speed * np.array([np.cos(heading), np.sin(heading)])
        pos = pos + vel * dt
        P.append(pos.copy())
        V.append(vel.copy())
    return np.array(P), np.array(V)


def _run_filter(filt, zs, dt=DT):
    est = []
    for z in zs:
        mu = filt.update(z, dt)[0]
        est.append(mu[:2])
    return np.array(est)


# ---------------- S-A1: IMM ----------------

def test_imm_rmse_beats_single_kf():
    """①合成 CV->CT 切换轨迹上, IMM 位置 RMSE < 单模型 CV-KF."""
    pos, vel = make_cv_ct_traj()
    rng = np.random.default_rng(0)
    zs = pos + rng.normal(0.0, 0.5, pos.shape)
    kw = dict(q=0.1, r=0.25)
    imm = IMMFilter(turn_rates=(0.0, -0.3, 0.3), **kw)
    kf = KFSingleCV(**kw)
    imm.reset(zs[0], vel[0])
    kf.reset(zs[0], vel[0])
    e_imm = _run_filter(imm, zs[1:]) - pos[1:]
    e_kf = _run_filter(kf, zs[1:]) - pos[1:]
    rmse_imm = float(np.sqrt((e_imm[10:] ** 2).sum(-1).mean()))
    rmse_kf = float(np.sqrt((e_kf[10:] ** 2).sum(-1).mean()))
    assert rmse_imm < rmse_kf, f"IMM {rmse_imm:.3f} 应低于 KF {rmse_kf:.3f}"


def test_imm_mode_switch_delay_under_1s():
    """②CV->CT 切换后, CT 模式合并概率过 0.5 的延迟 < 10 步(1 s)."""
    n_cv = 100
    pos, vel = make_cv_ct_traj(n_cv=n_cv, n_ct=60)
    rng = np.random.default_rng(1)
    zs = pos + rng.normal(0.0, 0.1, pos.shape)
    imm = IMMFilter(turn_rates=(0.0, -0.3, 0.3), q=0.05, r=0.01)
    imm.reset(zs[0], vel[0])
    ct_prob = []
    for z in zs[1:]:
        _, _, probs, _ = imm.update(z, DT)
        ct_prob.append(probs[1] + probs[2])   # omega=±0.3 两模型合并
    ct_prob = np.array(ct_prob)
    # 切换前稳态: CV 主导
    assert ct_prob[n_cv - 21:n_cv - 1].mean() < 0.5
    switch_idx = n_cv - 1                      # zs[1:] 下标系中首个 CT 步
    after = ct_prob[switch_idx:]
    detect = np.argmax(after > 0.5) if (after > 0.5).any() else len(after)
    assert detect < 10, f"模式切换检测延迟 {detect} 步 >= 10 步"


def test_imm_lost_trace_monotone_increase():
    """③失观期纯传播 predict: tr(Sigma) 单调增."""
    pos, vel = make_cv_ct_traj(n_cv=50, n_ct=0)
    imm = IMMFilter()
    imm.reset(pos[0], vel[0])
    for z in pos[1:]:
        imm.update(z, DT)
    traces = []
    for _ in range(50):
        _, Sigma, _ = imm.predict(DT)
        traces.append(np.trace(Sigma))
    traces = np.array(traces)
    assert (np.diff(traces) > 0).all(), "失观纯传播期 tr(Sigma) 应单调增"
    # 模式概率仍为合法分布
    assert abs(imm.mu_m.sum() - 1.0) < 1e-9 and (imm.mu_m >= 0).all()


# ---------------- S-A2: 单模型KF / 匀速外推 ----------------

def test_kf_single_tracks_and_smooths():
    """CV 轨迹上 KF 收敛且 RMSE 低于量测噪声(有效平滑)."""
    pos, vel = make_cv_ct_traj(n_cv=200, n_ct=0)
    rng = np.random.default_rng(2)
    sig = 0.5
    zs = pos + rng.normal(0.0, sig, pos.shape)
    kf = KFSingleCV(q=0.1, r=sig ** 2)
    kf.reset(zs[0], vel[0])
    err = _run_filter(kf, zs[1:]) - pos[1:]
    rmse = float(np.sqrt((err[20:] ** 2).sum(-1).mean()))
    noise_rms = float(np.sqrt(2.0) * sig)      # 2维量测噪声的位置RMS
    assert rmse < noise_rms, f"KF RMSE {rmse:.3f} 应低于量测噪声 {noise_rms:.3f}"


def test_kf_single_lost_trace_monotone():
    kf = KFSingleCV()
    kf.reset(np.array([10.0, 10.0]), np.array([2.0, 0.0]))
    kf.update(np.array([10.2, 10.0]), DT)
    traces = [np.trace(kf.predict(DT)[1]) for _ in range(30)]
    assert (np.diff(traces) > 0).all()


def test_cv_extrap_velocity_and_extrapolation():
    """两次精确观测后速度≈真值; 失观外推 = 线性外推; 协方差固定速率增长."""
    ext = CVExtrapolator(q_growth=4.0)
    v_true = np.array([3.0, -1.0])
    p0 = np.array([50.0, 50.0])
    ext.reset(p0)
    ext.update(p0 + v_true * DT, DT)
    mu, _, _, _ = ext.update(p0 + 2 * v_true * DT, DT)
    assert np.allclose(mu[2:], v_true, atol=1e-6), "有限差分速度应恢复真值"
    # 失观 10 步: 位置线性外推
    for k in range(1, 11):
        mu, Sigma, _ = ext.predict(DT)
        expect = p0 + (2 + k) * v_true * DT
        assert np.allclose(mu[:2], expect, atol=1e-9)
        assert np.isclose(Sigma[0, 0], ext.p0_pos + ext.q_growth * DT * k)
    # 重捕获: 协方差回落基准
    ext.update(mu[:2], DT)
    assert np.isclose(ext.P[0, 0], ext.p0_pos)


def test_interface_parity():
    """三个信念源接口同构: reset/update/predict 返回形状一致."""
    for filt in (IMMFilter(), KFSingleCV(), CVExtrapolator()):
        filt.reset(np.array([1.0, 2.0]), np.array([0.5, 0.0]))
        mu, Sigma, probs, lik = filt.update(np.array([1.1, 2.0]), DT)
        assert mu.shape == (4,) and Sigma.shape == (4, 4)
        assert probs.ndim == 1 and np.isfinite(lik) and lik > 0
        mu, Sigma, probs = filt.predict(DT)
        assert mu.shape == (4,) and Sigma.shape == (4, 4) and probs.ndim == 1
        assert np.isfinite(mu).all() and np.isfinite(Sigma).all()


def test_filter_manager_forwards_source_params():
    """训练/正式评估管理器必须把配置参数传入对应筛选器。"""
    from train_belief import FilterBeliefManager

    cases = [
        ("cv", {"q_growth": 16.0, "alpha": 0.8}, "q_growth", 16.0),
        ("kf", {"q": 8.0, "r": 0.01}, "q", 8.0),
        ("imm", {"q": 8.0, "r": 0.01, "p_stay": 0.95}, "q", 8.0),
    ]
    for source, kwargs, attr, expected in cases:
        mgr = FilterBeliefManager(source, "belief", 100.0, kwargs)
        assert mgr.filter_kwargs == kwargs
        assert getattr(mgr.filt, attr) == expected


def test_j4_belief_feature_variants_are_fixed_dimensional_and_orthogonal():
    """J4只屏蔽预注册通道，不改变均值或观测维数。"""
    from train_belief import select_belief_features

    sigma = np.array([0.1, 0.2, 0.03])
    conf = 0.37
    full = select_belief_features(sigma, conf, "belief")
    no_cov = select_belief_features(sigma, conf, "no_covariance")
    fixed_conf = select_belief_features(sigma, conf, "fixed_confidence")
    point = select_belief_features(sigma, conf, "point")
    assert np.array_equal(full[0], sigma) and full[1] == conf
    assert np.array_equal(no_cov[0], np.zeros(3)) and no_cov[1] == conf
    assert np.array_equal(fixed_conf[0], sigma) and fixed_conf[1] == 1.0
    assert np.array_equal(point[0], np.zeros(3)) and point[1] == 1.0


def test_j4_full_belief_path_preserves_prechange_bitwise_hash():
    """新增消融后，历史belief默认路径必须逐位不变。"""
    import hashlib

    from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv
    from train_belief import FilterBeliefManager

    cfg = EnvConfig(
        seed=20260729,
        n_dynamic_obs=2,
        obs_mode="belief",
        sense_range=25.0,
        fov_half=np.deg2rad(60.0),
    )
    env = PursuitLostEnv(cfg)
    manager = FilterBeliefManager(
        "imm", "belief", cfg.arena_size,
        {"q": 8.0, "r": 0.01, "p_stay": 0.95},
    )
    obs, state = env.reset(seed=20260729)
    obs = manager.reset(env)
    chunks = [obs.tobytes(), state.tobytes()]
    for step in range(40):
        action = np.zeros((cfg.n_pursuers, 2), dtype=float)
        action[:, 0] = 0.2 * np.sin(
            0.13 * step + np.arange(cfg.n_pursuers))
        action[:, 1] = 0.1 * np.cos(
            0.17 * step + np.arange(cfg.n_pursuers))
        obs, state, reward, done, _ = env.step(action)
        obs = manager.inject(env)
        chunks.extend([
            obs.tobytes(),
            state.tobytes(),
            np.asarray(reward).tobytes(),
            np.asarray(env.belief["mu"]).tobytes(),
            np.asarray(env.belief["sigma"]).tobytes(),
            np.asarray([env.belief["conf"]], dtype=np.float64).tobytes(),
        ])
        if done:
            break
    got = hashlib.sha256(b"".join(chunks)).hexdigest()
    # Transcendental/libm rounding differs between the original Windows host
    # and Linux server.  Preserve an exact digest for each verified platform
    # instead of silently weakening this regression to approximate equality.
    assert got in {
        "9ba70e1cb34d697ef1375c16b93f17105667a31249414f1caae20a3df20f4927",
        "2334ba6b679f5a21c129af15dd17d3eee61002384997a71ee91b54e637164112",
    }
