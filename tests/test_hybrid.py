import numpy as np
import torch

from belief.hybrid import (
    HYBRID_IN_DIM,
    HybridBeliefManager,
    IMMResidualGRU,
    hybrid_feature,
    load_hybrid_checkpoint,
)
from belief.imm_filter import IMMFilter
from scripts.eval_hybrid_residual import bootstrap_episode_differences
from scripts.summarize_hybrid_formal import retention_decision


def test_hybrid_feature_has_fixed_shape_and_no_lost_truth_leakage():
    filt = IMMFilter(q=8.0, r=0.01, p_stay=0.95)
    filt.reset(np.array([10.0, 20.0]), np.array([2.0, -1.0]))
    mu, cov, probs = filt.predict(0.1)
    left = hybrid_feature(np.array([99.0, 99.0]), np.array([7.0, 7.0]),
                          False, mu, cov, probs, 3, 100.0)
    right = hybrid_feature(np.array([-99.0, -99.0]), np.array([-7.0, -7.0]),
                           False, mu, cov, probs, 3, 100.0)
    assert left.shape == (HYBRID_IN_DIM,)
    assert np.isfinite(left).all()
    assert np.array_equal(left, right)


def test_residual_model_shapes_and_positive_variance():
    model = IMMResidualGRU(hidden=16)
    delta, var = model(torch.zeros(4, 20, HYBRID_IN_DIM))
    assert delta.shape == (4, 2)
    assert var.shape == (4, 2)
    assert torch.isfinite(delta).all()
    assert (var > 0).all()


def test_hybrid_checkpoint_round_trip(tmp_path):
    model = IMMResidualGRU(hidden=12)
    path = tmp_path / "hybrid.pt"
    torch.save({
        "model": model.state_dict(),
        "in_dim": HYBRID_IN_DIM,
        "hidden": 12,
        "seq_len": 20,
        "filter_params": {"q": 8.0, "r": 0.01, "p_stay": 0.95},
        "residual_temperature": 0.7,
    }, path)
    loaded, metadata = load_hybrid_checkpoint(str(path))
    assert loaded.hidden == 12
    assert metadata["residual_temperature"] == 0.7
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
        assert not actual.requires_grad


class _DummyConfig:
    arena_size = 100.0
    dt = 0.1


class _DummyEnv:
    def __init__(self):
        self.cfg = _DummyConfig()
        self.last_seen_pos = np.array([20.0, 30.0])
        self.t_pos = self.last_seen_pos.copy()
        self.t_vel = np.array([3.0, 1.0])
        self.visible_each = np.array([True, False, False])
        self.t_lost = 0
        self.belief = None

    def set_belief(self, mean, sigma, conf):
        self.belief = (np.asarray(mean), np.asarray(sigma), float(conf))

    def _get_obs(self):
        return np.concatenate([self.belief[0], self.belief[1],
                               [self.belief[2]]])


def test_hybrid_manager_uses_imm_visible_and_adds_lost_residual_covariance():
    model = IMMResidualGRU()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    env = _DummyEnv()
    manager = HybridBeliefManager(
        model, "belief", 100.0, residual_temperature=0.5,
        filter_kwargs={"q": 8.0, "r": 0.01, "p_stay": 0.95})
    manager.reset(env)
    visible_estimate = manager.last_estimate
    assert np.allclose(visible_estimate.mean, visible_estimate.imm_mean)
    env.visible_each[:] = False
    env.t_lost = 1
    manager.inject(env)
    lost_estimate = manager.last_estimate
    assert np.allclose(lost_estimate.mean, lost_estimate.imm_mean)
    assert np.all(np.diag(lost_estimate.covariance) >
                  np.diag(lost_estimate.imm_covariance))
    assert np.isfinite(env.belief[0]).all()
    assert 0.0 < env.belief[2] <= 1.0


def test_episode_cluster_bootstrap_is_paired_and_deterministic():
    details = []
    for episode, imm, hybrid in [(1, 2.0, 1.0), (2, 4.0, 3.0), (3, 6.0, 5.0)]:
        details.append(("imm", episode, 1, 1, imm, 0.0, 1.0))
        details.append(("hybrid", episode, 1, 1, hybrid, 0.0, 1.0))
    first = bootstrap_episode_differences(
        details, [("hybrid", "imm")], 200, 7)
    second = bootstrap_episode_differences(
        details, [("hybrid", "imm")], 200, 7)
    assert first == second
    assert first[0]["n_episodes"] == 3
    assert first[0]["mean_ade_difference_m"] == -1.0


def test_hybrid_retention_rule_requires_margin_and_seed_support():
    imm = {1: 0.10, 5: 0.10, 6: 0.10}
    retained = retention_decision(
        {1: 0.16, 5: 0.15, 6: 0.15}, imm)
    assert retained["retain_hybrid"]
    assert retained["matched_seed_support"] == 3

    insufficient_margin = retention_decision(
        {1: 0.14, 5: 0.14, 6: 0.14}, imm)
    assert not insufficient_margin["retain_hybrid"]

    insufficient_support = retention_decision(
        {1: 0.30, 5: 0.05, 6: 0.05}, imm)
    assert not insufficient_support["retain_hybrid"]
