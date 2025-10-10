# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations



from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.actuators import ImplicitActuatorCfg

# Import custom drone configuration
import os
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

# Forward declaration to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .betaflight_env import BetaflightEnv

# Define custom drone configuration
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
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

# Define custom drone with payload configuration
DRONE_WITH_PAYLOAD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "drone_with_payload.usda"),
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
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

class BetaflightEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: BetaflightEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)



@configclass
class BetaflightEnvCfg(DirectRLEnvCfg):

    payload = False

    # env
    episode_length_s = 10.0
    decimation = 2
    # Control input a0 in [-1, 1] that corresponds to hover (from Betaflight RC mid)
    hover_input: float = -0.547
    # Extra gain on thrust if needed (1.0 = none)
    thrust_gain: float = 1.0
    action_space = 4
    if payload:
        observation_space = 25
    else:
        observation_space = 19
    state_space = 0
    debug_vis = True

    ui_window_class_type = BetaflightEnvWindow

    # simulation
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    if payload:
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=DRONE_WITH_PAYLOAD_CFG.spawn,
            init_state=DRONE_WITH_PAYLOAD_CFG.init_state,
            actuators=DRONE_WITH_PAYLOAD_CFG.actuators,
        )
    else:
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=CUSTOM_DRONE_CFG.spawn,
            init_state=CUSTOM_DRONE_CFG.init_state,
            actuators=CUSTOM_DRONE_CFG.actuators,
        )
    moment_scale = 0.1

    # Angular velocity control parameters
    max_ang_vel_deg_s = 100.0  # Maximum angular velocity in degrees per second
    ang_vel_tau = 0.12  # First-order time constant for angular velocity response
    thrust_tau = 0.23 # First-order time constant for thrust response
    ang_vel_kp_roll_pitch = 0.1  # Proportional gain for roll and pitch angular velocity control
    ang_vel_kp_yaw = 0.1  # Proportional gain for yaw angular velocity control

    # Motor parameters for realistic thrust calculation
    max_motor_angular_vel = 4631.0  # Maximum motor angular velocity [rad/s]
    thrust_coefficient = 1.42e-06  # Thrust coefficient kt [N/(rad/s)^2]
    num_motors = 4  # Number of motors on the quadcopter

    # reward scales
    distance_threshold: float = 0.2

    if payload:
        # reward scales
        lin_vel_reward_scale: float = -0.05
        ang_vel_reward_scale: float = -0.01
        distance_to_goal_reward_scale: float = 15.0
        orientation_penalty_scale: float = -0.2
    else:
        lin_vel_reward_scale: float = -0.06
        ang_vel_reward_scale: float = -0.02
        distance_to_goal_reward_scale: float = 10.0
        orientation_penalty_scale: float = -0.5
    
    thrust_smoothness_penalty_scale: float = -0.2
    roll_smoothness_penalty_scale: float   = -0.1
    pitch_smoothness_penalty_scale: float  = -0.1
    yaw_smoothness_penalty_scale: float    = -0.1


    distance_normalizer: float = 0.8


    # Legacy parameters (kept for backward compatibility)
    thrust_ratio = 38.0  # Not used with motor-level thrust calculation
    mass = 0.6
    gravity = 9.81








