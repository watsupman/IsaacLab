# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym
import torch
import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers
from .betaflight_env_cfg import BetaflightEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class BetaflightEnv(DirectRLEnv):
    cfg: BetaflightEnvCfg

    def __init__(self, cfg: BetaflightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw = torch.zeros(self.num_envs, device=self.device)

        self._custom_goal = None
        self._custom_start = None
        self._goal_sequence = None
        self._goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Angular velocity control system variables
        self._desired_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)  # Desired angular velocities [rad/s]
        self._commanded_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)  # Current commanded angular velocities [rad/s]
        self._commanded_thrust = torch.zeros(self.num_envs, device=self.device)
        
        # Control system parameters from configuration
        self._max_ang_vel = math.pi * self.cfg.max_ang_vel_deg_s / 180.0  # Convert deg/s to rad/s
        self._ang_vel_tau = self.cfg.ang_vel_tau  # First-order time constant [s]
        self._thrust_tau = self.cfg.thrust_tau  # First-order time constant for thrust [s]

        self._prev_actions = torch.zeros_like(self._actions)

        # Logging
        keys = [
                "tracking",
                "lin_vel",
                "ang_vel",
                "orientation",
                "thrust_smoothness",
                "roll_smoothness",
                "pitch_smoothness",
                "yaw_smoothness",
            ]
        if getattr(self.cfg, "payload", False) and getattr(self.cfg, "payload_swing_reward_enable", False):
            keys += ["payload_swing_pos", "payload_swing_vel"]
        keys += ["total_reward"]

        self._episode_sums = {
        key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for key in keys
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._payload_id = int(self._robot.find_bodies("payload")[0][0]) if self.cfg.payload else None
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization
        # Per-env thrust constant (N) used to scale throttle->thrust
        # Sampled on reset during training; fixed to eval_thrust_constant otherwise.
        self._thrust_constant = torch.full((self.num_envs,), self.cfg.eval_thrust_constant, device=self.device)
        self.is_training = self.cfg.is_training
        # (this is set to a valid handle inside set_debug_vis)
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
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Motor-level thrust calculation

        
        # Calculate individual motor angular velocities from thrust command
        # Apply throttle exactly like Gazebo: (msg.channel_2 + 1.0)/2 * 4631
        # where msg.channel_2 is equivalent to self._actions[:, 0] (ranges from -1 to 1)

        # a0 = self._actions[:, 0].clamp(-1.0, 1.0)
        # u = (a0 + 1.0) * 0.5  # map [-1,1] -> [0,1]
        # u_hover = (self.cfg.hover_input + 1.0) * 0.5
        # u_hover = max(u_hover, 1e-3)  # avoid div-by-zero
        # hover_thrust = self._robot_weight  # N
        # scale = hover_thrust / u_hover  # N per unit u so that u=u_hover gives hover
        # desired_thrust = self.cfg.thrust_gain * scale * u

        force = (self._actions[:, 0] + 1.0) / 2.0 
        
        desired_thrust = self._thrust_constant * force

        dt = self.cfg.sim.dt * self.cfg.decimation
        alpha = dt / (self._thrust_tau + dt)
        self._commanded_thrust = (1.0 - alpha) * self._commanded_thrust + alpha * desired_thrust


        # Apply thrust in body Z direction (up)
        self._thrust[:, 0, 0] = 0.0  # No X thrust
        self._thrust[:, 0, 1] = 0.0  # No Y thrust  
        self._thrust[:, 0, 2] = self._commanded_thrust # total_thrust /self._robot_mass  # Z thrust (upward)

        #print(f"thrust {self._actions[:, 0]} mass {self._robot_mass}")


        # Process angular velocity commands (unchanged)
        # actions[1:4] represent desired angular velocities: [roll_rate, pitch_rate, yaw_rate]
        # Map from [-1, 1] to [-max_ang_vel, max_ang_vel]
        desired_ang_vel = self._actions[:, 1:4] * self._max_ang_vel
        
        # Invert yaw command (yaw is inverted)
        desired_ang_vel[:, 2] *= -1.0
        
        self._desired_ang_vel = desired_ang_vel
        
        # First-order system response: τ * dω/dt + ω = ω_desired
        # Discrete form: ω[k+1] = ω[k] + dt/τ * (ω_desired - ω[k])
        dt = self.cfg.sim.dt * self.cfg.decimation  # Control timestep
        alpha = dt / (self._ang_vel_tau + dt)  # Filter coefficient
        
        self._commanded_ang_vel = (1.0 - alpha) * self._commanded_ang_vel + alpha * self._desired_ang_vel
        
        # Get current angular velocity in body frame
        current_ang_vel = self._robot.data.root_ang_vel_b
        
        # Calculate angular velocity error
        ang_vel_error = self._commanded_ang_vel - current_ang_vel
        
        # Convert angular velocity error to moments using proportional control
        # Use gains from configuration
        kp_roll_pitch = self.cfg.ang_vel_kp_roll_pitch  # Proportional gain for roll and pitch
        kp_yaw = self.cfg.ang_vel_kp_yaw                # Proportional gain for yaw
        
        self._moment[:, 0, 0] = kp_roll_pitch * ang_vel_error[:, 0]   # Roll moment
        self._moment[:, 0, 1] = kp_roll_pitch * ang_vel_error[:, 1]   # Pitch moment  
        self._moment[:, 0, 2] = kp_yaw * ang_vel_error[:, 2]          # Yaw moment

    def _apply_action(self) -> None:
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
    
    def _get_observations(self) -> dict:
        quad_pos_w = self._robot.data.root_state_w[:, :3]
        quad_quat_w = self._robot.data.root_state_w[:, 3:7]

        desired_pos_b, _ = subtract_frame_transforms(
            quad_pos_w,
            quad_quat_w,
            self._desired_pos_w
        )
        q = quad_quat_w  # [w, x, y, z]
        w, x, y, z = q.unbind(-1)
        curr_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_err = torch.atan2(torch.sin(curr_yaw - self._desired_yaw),
                            torch.cos(curr_yaw - self._desired_yaw))
        heading_error = torch.stack([torch.sin(yaw_err), torch.cos(yaw_err)], dim=-1)
        
        if self._payload_id is not None:
            payload_pos_w = self._robot.data.body_pos_w[:, self._payload_id, :]
            payload_vel_w = self._robot.data.body_lin_vel_w[:, self._payload_id, :]
            relative_payload_pos_b, _ = subtract_frame_transforms(
                quad_pos_w, quad_quat_w, payload_pos_w
            )
            # relative_payload_vel_b = payload_vel_w - self._robot.data.root_lin_vel_w
            v_rel_w = payload_vel_w - self._robot.data.root_lin_vel_w
            R_wb = self.quat_wxyz_to_rotmat(quad_quat_w)
            relative_payload_vel_b = torch.bmm(R_wb, v_rel_w.unsqueeze(-1)).squeeze(-1)
            obs = torch.cat([
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                quad_quat_w,
                desired_pos_b,
                heading_error,
                self._prev_actions, 
                relative_payload_pos_b,
                relative_payload_vel_b,
            ], dim=-1)
        else:
                obs = torch.cat([
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                quad_quat_w,
                desired_pos_b,
                heading_error,
                self._prev_actions,
            ], dim=-1)
        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_reward = 1.0 - torch.tanh(distance_to_goal / self.cfg.distance_normalizer)
        distance_scale = self.cfg.distance_to_goal_reward_scale
        
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        q = self._robot.data.root_state_w[:, 3:7]  # [w,x,y,z]
        w, x, y, z = q.unbind(-1)
        curr_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        yaw_err = torch.atan2(torch.sin(curr_yaw - self._desired_yaw),
                            torch.cos(curr_yaw - self._desired_yaw))

        heading_pen = (1 - torch.cos(yaw_err)) * self.cfg.orientation_penalty_scale * self.step_dt

        action_diff = self._actions - self._prev_actions
        thrust_rate = action_diff[:, 0] ** 2
        roll_rate   = action_diff[:, 1] ** 2
        pitch_rate  = action_diff[:, 2] ** 2
        yaw_rate    = action_diff[:, 3] ** 2

        thrust_smooth_penalty = thrust_rate * self.cfg.thrust_smoothness_penalty_scale * self.step_dt
        roll_smooth_penalty   = roll_rate   * self.cfg.roll_smoothness_penalty_scale   * self.step_dt
        pitch_smooth_penalty  = pitch_rate  * self.cfg.pitch_smoothness_penalty_scale  * self.step_dt
        yaw_smooth_penalty    = yaw_rate    * self.cfg.yaw_smoothness_penalty_scale    * self.step_dt


        rewards = {
            "tracking": distance_reward * distance_scale * self.step_dt,
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "orientation": heading_pen,
            "thrust_smoothness": thrust_smooth_penalty,
            "roll_smoothness": roll_smooth_penalty,
            "pitch_smoothness": pitch_smooth_penalty,
            "yaw_smoothness": yaw_smooth_penalty,
        }


        # --- Payload swing minimization (conditional) ---
        if self._payload_id is not None and getattr(self.cfg, "payload_swing_reward_enable", False):
            payload_state = self.get_payload_state()
            if payload_state is not None:
                rel_pos_b, rel_vel_b = payload_state
                # Penalize lateral (X/Y) displacement and velocity to reduce swing
                pos_xy_sq = torch.sum(rel_pos_b[:, :2] ** 2, dim=1)
                vel_xy_sq = torch.sum(rel_vel_b[:, :2] ** 2, dim=1)
                rewards["payload_swing_pos"] = pos_xy_sq * self.cfg.payload_swing_pos_penalty_scale * self.step_dt
                rewards["payload_swing_vel"] = vel_xy_sq * self.cfg.payload_swing_vel_penalty_scale * self.step_dt

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["total_reward"] += reward

        if self._goal_sequence is not None and self._custom_goal is None:
            goal_reached = torch.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < self.cfg.distance_threshold
            for i in range(self.num_envs):
                if goal_reached[i]:
                    self._goal_idx[i] = (self._goal_idx[i] + 1) % len(self._goal_sequence)
                    self._desired_pos_w[i] = self._goal_sequence[self._goal_idx[i]]

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 3.0)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        #else:
            # log the rewards
            #for key, value in self._episode_sums.items():
                #print(f"Episode finished! {key}: {value[env_ids].mean().item()}")

        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

        self._actions[env_ids] = 0.0
        # Re/initialize per-env thrust constant
        if self.cfg.thrust_constant_train_only and self.is_training:
            # Gaussian sample, then clamp to [min, max]
            mean = self.cfg.thrust_constant_gauss_mean
            std = self.cfg.thrust_constant_gauss_std
            samples = torch.normal(mean=torch.full((len(env_ids),), mean, device=self.device),
                                   std=torch.full((len(env_ids),), std, device=self.device))
            samples = samples.clamp(self.cfg.thrust_constant_clip_min, self.cfg.thrust_constant_clip_max)
            self._thrust_constant[env_ids] = samples
        else:
            self._thrust_constant[env_ids] = self.cfg.eval_thrust_constant

        # Reset angular velocity control states
        self._desired_ang_vel[env_ids] = 0.0
        self._commanded_ang_vel[env_ids] = 0.0

        self._desired_yaw[env_ids] = 0.0
        
        # Sample new commands
        if self._custom_goal is not None:
            self._desired_pos_w[env_ids] = self._custom_goal.expand(len(env_ids), -1)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        elif self._goal_sequence is not None:
            self._goal_idx[env_ids] = 0
            self._desired_pos_w[env_ids] = self._goal_sequence[self._goal_idx[env_ids]].expand(len(env_ids), -1)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        else:
            # Randomized for training
            self._desired_pos_w[env_ids, :2] = (
                torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
                + self._terrain.env_origins[env_ids, :2]
            )
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 2.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        if self._custom_start is not None:
            default_root_state[:, :3] = self._custom_start.expand(len(env_ids), -1)
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = copy.deepcopy(CUBOID_MARKER_CFG)
                marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
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

    @staticmethod
    def quat_wxyz_to_rotmat(q):
        # q: (N,4) or (4,)
        if not torch.is_tensor(q):
            q = torch.as_tensor(q, dtype=torch.float32)
        q = q.to(dtype=torch.float32)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        # normalize
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
        w, x, y, z = q.unbind(-1)

        # Build R_bw (body->world)
        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y - w * z)
        r02 = 2.0 * (x * z + w * y)

        r10 = 2.0 * (x * y + w * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z - w * x)

        r20 = 2.0 * (x * z - w * y)
        r21 = 2.0 * (y * z + w * x)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        R_bw = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=-2,
        )
        # Return R_wb so v_b = R_wb @ v_w
        return R_bw.transpose(-1, -2)

