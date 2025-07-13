# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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

import isaacsim.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.usd
from pxr import Gf, Usd, UsdPhysics, UsdGeom, PhysxSchema

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG, DRONE_WITH_PAYLOAD_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
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
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    trajectory: bool = False
    payload: bool = True
    payload_aware: bool = False

    episode_length_s: float = 20.0
    decimation: int = 2
    action_space: int = 4
    if payload_aware:
        observation_space: int = 18
    else:
        observation_space: int = 12
    state_space: int = 0
    debug_vis: bool = True

    ui_window_class_type = QuadcopterEnvWindow

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
    robot: ArticulationCfg = DRONE_WITH_PAYLOAD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # thrust_to_weight = 1.9
    thrust_to_weight = 20.7
    moment_scale = 0.01

    distance_threshold = 0.2

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale_fixed = 15.0
    distance_to_goal_reward_scale_traj = 15.0

    distance_normalizer_fixed = 0.8
    distance_normalizer_traj = 0.5


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._custom_goal = None
        self._custom_start = None
        self._goal_sequence = None
        self._goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tracking",
                "lin_vel",
                "ang_vel",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._payload_id = int(self._robot.find_bodies("payload")[0][0]) if self.cfg.payload else None
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def set_custom_goal(self, goal_tensor):
        self._custom_goal = goal_tensor.clone().to(self.device)

    def set_custom_start(self, start_tensor):
        self._custom_start = start_tensor.clone().to(self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

    
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):

        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]


    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            self._thrust[:self.num_envs],
            self._moment[:self.num_envs],
            body_ids=self._body_id
        )

    def _get_observations(self) -> dict:
        quad_pos_w = self._robot.data.root_state_w[:, :3]
        quad_quat_w = self._robot.data.root_state_w[:, 3:7]

        desired_pos_b, _ = subtract_frame_transforms(
            quad_pos_w,
            quad_quat_w,
            self._desired_pos_w
        )
        if self._payload_id is not None:
            payload_pos_w = self._robot.data.body_pos_w[:, self._payload_id, :]
            payload_vel_w = self._robot.data.body_lin_vel_w[:, self._payload_id, :]
            relative_payload_pos_b, _ = subtract_frame_transforms(
                quad_pos_w, quad_quat_w, payload_pos_w
            )
            relative_payload_vel_b = payload_vel_w - self._robot.data.root_lin_vel_w
        if self.cfg.payload_aware and self._payload_id is not None:
            obs = torch.cat([
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                relative_payload_pos_b,
                relative_payload_vel_b,
            ], dim=-1)
        else:
                obs = torch.cat([
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ], dim=-1)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_reward = 1.0 - torch.tanh(distance_to_goal / self.cfg.distance_normalizer_fixed)
        distance_scale = self.cfg.distance_to_goal_reward_scale_fixed
        
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        rewards = {
            "tracking": distance_reward * distance_scale * self.step_dt,
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        if self._goal_sequence is not None and self._custom_goal is None:
            goal_reached = torch.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < self.cfg.distance_threshold
            for i in range(self.num_envs):
                if goal_reached[i]:
                    self._goal_idx[i] = (self._goal_idx[i] + 1) % len(self._goal_sequence)
                    self._desired_pos_w[i] = self._goal_sequence[self._goal_idx[i]]

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 5.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging and standard reset logic ...

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # Custom or random goal
        if self._custom_goal is not None:
            self._desired_pos_w[env_ids] = self._custom_goal.expand(len(env_ids), -1)
        elif self._goal_sequence is not None:
            self._goal_idx[env_ids] = 0
            self._desired_pos_w[env_ids] = self._goal_sequence[self._goal_idx[env_ids]].expand(len(env_ids), -1)
        else:
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # Custom or default starting position
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        if self._custom_start is not None:
            default_root_state[:, :3] = self._custom_start.expand(len(env_ids), -1)
        else:
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            self.goal_pos_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def set_goal_sequence(self, goal_list: list[torch.Tensor]):
        self._goal_sequence = [g.to(self.device) for g in goal_list]
        self._goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._desired_pos_w = torch.stack([g for g in goal_list[:self.num_envs]]).to(self.device)

    def get_payload_state(self):
        if self._payload_id is None:
            return None
        pos = self._robot.data.body_pos_w[:, self._payload_id, :]
        vel = self._robot.data.body_lin_vel_w[:, self._payload_id, :]
        quad_pos_w = self._robot.data.root_state_w[:, :3]
        quad_quat_w = self._robot.data.root_state_w[:, 3:7]
        relative_payload_pos_b, _ = subtract_frame_transforms(
            quad_pos_w, quad_quat_w, pos
        )
        relative_payload_vel_b = vel - self._robot.data.root_lin_vel_w
        return relative_payload_pos_b, relative_payload_vel_b
