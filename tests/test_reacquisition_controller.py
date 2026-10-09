import math
from types import SimpleNamespace

import numpy as np

from utils.reacquisition import ReacquisitionController, build_reacquisition_controller


def _env(obs_mode="last"):
    return SimpleNamespace(
        n=3,
        cfg=SimpleNamespace(
            obs_mode=obs_mode,
            arena_size=100.0,
            v_max=8.0,
            a_max=4.0,
            w_max=1.5,
            lost_cap_steps=100,
        ),
        p_pos=np.array([[40.0, 50.0], [10.0, 10.0], [80.0, 80.0]]),
        p_psi=np.array([0.0, math.pi / 2.0, -math.pi / 2.0]),
        p_v=np.array([2.0, 2.0, 2.0]),
        visible_each=np.array([False, False, False]),
        t_lost=5,
        last_seen_pos=np.array([50.0, 50.0]),
        belief=None,
        # Deliberately different: controller must not use hidden target truth.
        t_pos=np.array([99.0, 1.0]),
    )


def test_disabled_factory_preserves_existing_behavior():
    assert build_reacquisition_controller(None) is None
    assert build_reacquisition_controller({"enabled": False}) is None


def test_visible_step_does_not_override_policy_action():
    env = _env()
    env.visible_each[1] = True
    controller = ReacquisitionController({"enabled": True})
    policy = np.full((3, 2), 0.2)
    composed, diagnostics = controller.apply(env, policy)
    np.testing.assert_allclose(composed, policy)
    assert diagnostics["active"] is False


def test_loss_uses_last_seen_reference_and_assigns_nearest_scout():
    env = _env("last")
    controller = ReacquisitionController({"enabled": True})
    policy = np.zeros((3, 2))
    composed, diagnostics = controller.apply(env, policy)
    assert diagnostics["active"] is True
    assert diagnostics["reference_source"] == "last_seen"
    assert diagnostics["center_x"] == 50.0
    assert diagnostics["center_y"] == 50.0
    assert diagnostics["scout_index"] == 0
    assert composed.shape == (3, 2)
    assert np.isfinite(composed).all()
    assert np.abs(composed).max() <= 1.0
    assert not np.allclose(composed, policy)
    assert len(np.unique(np.round(controller.slot_angles, 6))) == 3


def test_belief_reference_is_used_when_available():
    env = _env("belief")
    env.belief = {
        "mu": np.array([25.0, 75.0]),
        "sigma": np.ones(3),
        "conf": 0.5,
    }
    controller = ReacquisitionController({"enabled": True})
    _, diagnostics = controller.apply(env, np.zeros((3, 2)))
    assert diagnostics["reference_source"] == "belief"
    assert diagnostics["center_x"] == 25.0
    assert diagnostics["center_y"] == 75.0


def test_reacquisition_resets_role_when_visibility_returns():
    env = _env()
    controller = ReacquisitionController({"enabled": True})
    controller.apply(env, np.zeros((3, 2)))
    assert controller.active
    env.visible_each[2] = True
    policy = np.full((3, 2), -0.1)
    composed, diagnostics = controller.apply(env, policy)
    np.testing.assert_allclose(composed, policy)
    assert diagnostics["active"] is False
    assert controller.active is False
    assert controller.scout_index is None


def test_post_reacquisition_hold_contracts_slots_before_release():
    env = _env()
    controller = ReacquisitionController({
        "enabled": True,
        "post_reacquisition_hold_steps": 3,
        "contraction_steps": 2,
        "slot_radius_m": 8.0,
        "capture_slot_radius_m": 4.0,
        "visible_tracker_blend": 0.7,
    })
    policy = np.zeros((3, 2))
    controller.apply(env, policy)
    assert controller.active

    env.visible_each[0] = True
    first, first_diag = controller.apply(env, policy)
    assert first_diag["phase"] == "visible_contraction"
    assert first_diag["slot_radius_m"] == 6.0
    assert controller.active
    assert not np.allclose(first, policy)

    _, second_diag = controller.apply(env, policy)
    assert second_diag["slot_radius_m"] == 4.0
    assert controller.active

    _, third_diag = controller.apply(env, policy)
    assert third_diag["phase"] == "visible_contraction"
    assert controller.active is False

    released, released_diag = controller.apply(env, policy)
    np.testing.assert_allclose(released, policy)
    assert released_diag["active"] is False


def test_invalid_contraction_parameters_are_rejected():
    with np.testing.assert_raises(ValueError):
        ReacquisitionController({
            "slot_radius_m": 8.0,
            "capture_slot_radius_m": 9.0,
        })


def test_always_on_visible_geometry_uses_observed_reference():
    env = _env()
    env.visible_each[0] = True
    controller = ReacquisitionController({
        "enabled": True,
        "always_on_visible": True,
        "visible_tracker_blend": 1.0,
        "capture_slot_radius_m": 4.0,
    })
    policy = np.zeros((3, 2))
    composed, diagnostics = controller.apply(env, policy)
    assert diagnostics["active"] is True
    assert diagnostics["phase"] == "visible_geometric"
    assert diagnostics["reference_source"] == "last_seen"
    assert diagnostics["center_x"] == 50.0
    assert diagnostics["center_y"] == 50.0
    assert not np.allclose(composed, policy)
    assert controller.active is True
