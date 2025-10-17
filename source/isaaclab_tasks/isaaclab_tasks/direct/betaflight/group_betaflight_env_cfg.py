# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

# Forward declaration to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .group_betaflight_env import BetaflightEnv

# ---------- Base drone configs ----------
CUSTOM_DRONE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "custom_drone.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={
            "m1_joint": 200.0, "m2_joint": -200.0, "m3_joint": 200.0, "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
    },
)


class BetaflightEnvWindow(BaseEnvWindow):
    def __init__(self, env: BetaflightEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class BetaflightEnvCfg(DirectRLEnvCfg):
    """Config for single- or multi-drone (group) quadcopter env.

    Set ``num_robots`` to the number of drones (>=1).
    A regular N-gon formation is used for spawn when ``num_robots > 1``.
    """

    # --------- Modes ---------
    num_robots: int = 4
    group_mode: bool = True          # kept for backward-compat; ignored at runtime if num_robots is set
    is_training: bool = False        # False for eval/inference

    # --------- Thrust constant randomization (training-only) ---------
    thrust_constant_train_only: bool = False
    thrust_constant_gauss_mean: float = 38.0
    thrust_constant_gauss_std: float = 1.5
    thrust_constant_clip_min: float = 36.0
    thrust_constant_clip_max: float = 42.0
    eval_thrust_constant: float = 38.0

    # --------- Env timing / spaces ---------
    episode_length_s = 10.0
    decimation = 2
    action_space = 4 * num_robots          # updated by env at runtime as well
    # Observation dims:
    # - Single: 3 + 3 + 4 + 3 + 2 + 4 = 19
    # - Group (N): baseline 14 per robot; env refines at runtime to include neighbor terms
    observation_space = (14 * num_robots + 5) if num_robots > 1 else 19
    state_space = 0
    debug_vis = True

    ui_window_class_type = BetaflightEnvWindow

    # --------- Simulation ---------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --------- Scene ---------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=8.5, replicate_physics=True)

    # --------- Robot (single config reused for all drones) ---------
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=CUSTOM_DRONE_CFG.spawn,
        init_state=CUSTOM_DRONE_CFG.init_state,
        actuators=CUSTOM_DRONE_CFG.actuators,
    )

    # --------- Control params ---------
    moment_scale = 0.1
    max_ang_vel_deg_s = 100.0
    ang_vel_tau = 0.12
    thrust_tau = 0.18
    ang_vel_kp_roll_pitch = 0.1
    ang_vel_kp_yaw = 0.1

    # Motor-ish params (kept for compatibility; thrust_constant is used)
    max_motor_angular_vel = 4631.0
    thrust_coefficient = 1.42e-06
    num_motors = 4

    # --------- Rewards / penalties ---------
    distance_threshold: float = 0.2
    lin_vel_reward_scale: float = -0.005
    ang_vel_reward_scale: float = -0.02
    distance_to_goal_reward_scale: float = 11.0
    orientation_penalty_scale: float = -0.2
    thrust_smoothness_penalty_scale: float = -0.2
    roll_smoothness_penalty_scale: float   = -0.1
    pitch_smoothness_penalty_scale: float  = -0.1
    yaw_smoothness_penalty_scale: float    = -0.1
    distance_normalizer: float = 0.8

    # Hover input legacy
    hover_input: float = -0.54
    thrust_gain: float = 1.0

    # Legacy ratio/mass (unused for dynamics but retained for compatibility)
    thrust_ratio = 38.0
    mass = 0.6
    gravity = 9.81

    # Encourage same altitude across the formation (variance of z across drones)
    z_alignment_penalty_scale: float = -0.5  # negative -> penalize variance; increase magnitude for stronger coupling

    # --------- Formation specifics (group mode) ---------
    neighbors_k: int = 2               # number of nearest neighbors to include per drone in observations
    # formation_target_distance: float = 0.56    # side length of the regular N-gon (adjacent drones)
    min_separation: float = 0.15             # hard safety radius
    collision_penalty_scale: float =  -8.0
    formation_target_distance: float = 0.56

    if num_robots == 3:
        distance_to_goal_reward_scale: float = 10.0
        z_alignment_penalty_scale: float = -1.5
        formation_separation_penalty_scale: float = -7.0
        ring_radial_penalty_scale: float = -3.5
        neighbor_velocity_penalty_scale: float = -0.5
        center_velocity_penalty_scale: float = -0.2
    else:
        formation_separation_penalty_scale: float = -2.0
        ring_radial_penalty_scale: float = -1.5
        neighbor_velocity_penalty_scale: float = -0.5
        center_velocity_penalty_scale: float = -0.2

    # --------- NEW: Max separation constraint ---------
    # If any inter-drone distance exceeds this, apply a large penalty (per step).
    # You can also enable hard termination below.
    max_pairwise_separation: float = 0.8  # meters
    max_separation_penalty_scale: float = -50.0  # strong negative when violated (applied with step_dt)
    terminate_on_max_separation_exceeded: bool = True  # set True to end the episode immediately


    # --------- NEW: Collision termination ---------
    # If any pair of drones comes closer than ``min_separation``, end the episode.
    terminate_on_collision: bool = True
