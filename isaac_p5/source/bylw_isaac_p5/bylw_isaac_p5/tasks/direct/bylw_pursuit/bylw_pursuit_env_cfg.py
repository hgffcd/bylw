"""Isaac Lab 2.1 configuration for the P5 fixed-height task."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG


STATIC_OBSTACLE_SPECS = (
    (-18.0, -15.0, 3.0),
    (-18.0, 15.0, 3.0),
    (-4.0, -18.0, 4.0),
    (-4.0, 18.0, 4.0),
    (10.0, -15.0, 3.5),
    (10.0, 15.0, 3.5),
    (25.0, -12.0, 3.0),
    (25.0, 12.0, 3.0),
)
STATIC_OBSTACLE_SHAPES = ("cylinder",) * len(STATIC_OBSTACLE_SPECS)
STATIC_OBSTACLE_BOX_SIZES_XY_M = ((0.0, 0.0),) * len(STATIC_OBSTACLE_SPECS)

# D1/C13 retains the authored 3--4 m cylinder scale and adds four physical
# bodies.  Episode reset still randomizes every centre, so these coordinates
# only provide a separated scene-construction pose and do not define a map.
DENSE_STATIC_OBSTACLE_SPECS = STATIC_OBSTACLE_SPECS + (
    (-30.0, 0.0, 3.0),
    (30.0, 0.0, 3.0),
    (0.0, -30.0, 3.5),
    (0.0, 30.0, 3.5),
)
DENSE_STATIC_OBSTACLE_SHAPES = ("cylinder",) * len(DENSE_STATIC_OBSTACLE_SPECS)
DENSE_STATIC_OBSTACLE_BOX_SIZES_XY_M = (
    (0.0, 0.0),
) * len(DENSE_STATIC_OBSTACLE_SPECS)

# C14 changes only physical obstacle shape/size diversity.  The first six
# bodies are cylinders with 2--5 m radii.  The remaining six are square or
# rectangular columns.  For each box, obstacle_specs stores the circumscribed
# circle radius used by perception, reset separation, and the predictive
# safety layer; Isaac physics still uses the exact CuboidCfg dimensions below.
MIXED_STATIC_OBSTACLE_BOX_SIZES_XY_M = (
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (4.0, 8.0),
    (6.0, 6.0),
    (4.0, 6.0),
    (5.0, 8.0),
    (4.0, 5.0),
    (6.0, 5.0),
)
MIXED_STATIC_OBSTACLE_SPECS = (
    (-18.0, -15.0, 2.0),
    (-18.0, 15.0, 2.5),
    (-4.0, -18.0, 3.0),
    (-4.0, 18.0, 3.5),
    (10.0, -15.0, 4.0),
    (10.0, 15.0, 5.0),
    (25.0, -12.0, 4.47213595),
    (25.0, 12.0, 4.24264069),
    (-30.0, 0.0, 3.60555128),
    (30.0, 0.0, 4.71699057),
    (0.0, -30.0, 3.20156212),
    (0.0, 30.0, 3.90512484),
)
MIXED_STATIC_OBSTACLE_SHAPES = (
    ("cylinder",) * 6 + ("box",) * 6
)


@configclass
class BylwPursuitEnvCfg(DirectMARLEnvCfg):
    # 100 Hz physics/controller, ten substeps per 10 Hz frozen actor action.
    decimation = 10
    episode_length_s = 50.0
    possible_agents = ["pursuer_0", "pursuer_1", "pursuer_2"]
    # P5-0/P5-1 remain three-agent by default. P5-C opts in to variable
    # scale before environment construction; the extra assets are declared
    # here so Isaac can clone them without changing the default path.
    n_pursuers = 3
    min_capture_agents = 3
    allow_variable_scale = False
    action_spaces = {agent: 2 for agent in possible_agents}
    observation_spaces = {agent: 39 for agent in possible_agents}
    state_space = 32
    seed = 20260824

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=120.0,
        replicate_physics=True,
    )

    pursuer_0_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Pursuer_0"
    )
    pursuer_1_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Pursuer_1"
    )
    pursuer_2_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Pursuer_2"
    )
    pursuer_3_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Pursuer_3"
    )
    pursuer_4_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Pursuer_4"
    )
    evader_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Evader"
    )

    runtime_gate_env_var = "BYLW_P5_ENABLE_ISAAC_RUNTIME"
    fixed_altitude_m = 3.0
    arena_size_m = 100.0
    sense_range_m = 25.0
    fov_half_rad = math.pi / 3.0
    capture_radius_main_m = 6.0
    capture_radius_strict_m = 5.0
    max_capture_gap_rad = math.pi
    contact_radius_m = 0.3
    frame_accurate_loss_counters = False
    pursuer_radius_m = 0.5
    # P5-1 formal evidence remains on legacy_fixed_cluster.  tracked_ring is
    # opt-in for P5-1R and matches the frozen 2-D tracked-start reset contract.
    reset_layout = "legacy_fixed_cluster"
    tracked_target_center_xy_m = (35.0, 0.0)
    tracked_target_spread_m = 6.0
    tracked_safe_agent_distance_m = 2.5
    # Optional reference range used only to sample tracked-ring reset geometry.
    # Keeping this separate from sense_range_m permits paired sensor ablations
    # without silently moving the initial pursuers closer to the target.
    tracked_reset_reference_range_m = None
    # C96: training-only; formal evaluation forcibly disables this switch.
    near_ring_reset_curriculum_enabled = False
    dense_physical_telemetry = False
    evader_enabled = True
    evader_speed_mps = 4.8
    # Named capability limits make equal/stronger-evader tests auditable.
    # Their defaults preserve the two controllers' previous numerical values.
    pursuer_max_planar_accel_mps2 = 4.0
    evader_max_planar_accel_mps2 = 4.0
    evader_minimum_episode_speed_mps = 0.0
    evader_dwell_min_s = 2.0
    evader_dwell_max_s = 4.0
    evader_turn_rate_rps = 0.3
    evader_stop_tau_s = 0.3
    evader_pursuer_repulsion_gain = 1.0
    evader_pursuer_reaction_range_m = 30.0
    evader_allow_stop_mode = True
    evader_gap_escape_weight = 0.0
    evader_maintain_episode_speed = False
    invalid_height_tolerance_m = 2.0
    obstacle_height_m = 8.0
    # G1 opt-in domain randomization.  Each episode receives a deterministic
    # irregular physical layout derived from its episode seed; the layout is
    # held fixed for the full episode.  Frozen P5-0/P5-1 paths remain disabled.
    randomize_obstacles_on_reset = False
    obstacle_sampling_half_extent_m = 32.0
    obstacle_minimum_surface_gap_m = 2.0
    obstacle_central_keepout_m = 5.0
    # Empty for all frozen stages.  C16 enables two kinematic obstacles whose
    # seeded planar velocity reverses periodically, keeping motion bounded.
    dynamic_obstacle_indices = ()
    dynamic_obstacle_speed_mps = 0.6
    dynamic_obstacle_reversal_interval_s = 2.0
    # Deterministic P5-0B map. Random unseen maps are added only after smoke.
    obstacle_specs = STATIC_OBSTACLE_SPECS
    obstacle_shape_kinds = STATIC_OBSTACLE_SHAPES
    obstacle_box_sizes_xy_m = STATIC_OBSTACLE_BOX_SIZES_XY_M
