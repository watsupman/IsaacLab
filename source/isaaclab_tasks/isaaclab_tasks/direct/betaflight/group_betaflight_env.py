from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym
import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .group_betaflight_env_cfg import BetaflightEnvCfg


class BetaflightEnv(DirectRLEnv):
    cfg: BetaflightEnvCfg

    def __init__(self, cfg: BetaflightEnvCfg, render_mode: str | None = None, **kwargs):
        # ----- Determine number of robots / group mode early (DirectRLEnv.__init__ uses spaces) -----
        num_from_cfg = int(getattr(cfg, "num_robots", 2 if getattr(cfg, "group_mode", False) else 1))
        self._num_robots = max(1, num_from_cfg)
        self.group_mode = self._num_robots > 1
        # Keep cfg in sync (spaces used by base class)
        cfg.group_mode = self.group_mode
        cfg.action_space = 4 * self._num_robots
        # Nearest-neighbor features per robot.
        # Clamp K so we never include the robot itself as a neighbor when N is small.
        neighbors_k_cfg = int(getattr(cfg, "neighbors_k", 2))
        self._neighbors_k_eff = max(0, min(neighbors_k_cfg, self._num_robots - 1))
        per_robot_obs = 14 + 3 * self._neighbors_k_eff
        cfg.observation_space = (per_robot_obs * self._num_robots + 5) if self.group_mode else 19

        # Now let the base class set up the scene, device, num_envs, etc.
        super().__init__(cfg, render_mode, **kwargs)
        # ---- Validate required config fields (fail-fast with clear error) ----
        _required_cfg_fields = [
            "max_ang_vel_deg_s", "ang_vel_tau", "thrust_tau",
            "ang_vel_kp_roll_pitch", "ang_vel_kp_yaw",
            "eval_thrust_constant", "is_training",
            # Rewards / penalties
            "distance_normalizer", "distance_to_goal_reward_scale",
            "lin_vel_reward_scale", "ang_vel_reward_scale",
            "orientation_penalty_scale",
            "thrust_smoothness_penalty_scale", "roll_smoothness_penalty_scale",
            "pitch_smoothness_penalty_scale", "yaw_smoothness_penalty_scale",
            # Group / formation
            "neighbors_k", "formation_target_distance",
            "min_separation", "formation_separation_penalty_scale",
            "collision_penalty_scale", "ring_radial_penalty_scale",
            "neighbor_velocity_penalty_scale", "center_velocity_penalty_scale",
            "z_alignment_penalty_scale",
            # Max separation constraint (penalty + optional termination)
            "max_pairwise_separation", "max_separation_penalty_scale",
            "terminate_on_max_separation_exceeded",
            "terminate_on_collision",

            # Misc
            "debug_vis",
        ]
        _missing = [k for k in _required_cfg_fields if not hasattr(self.cfg, k)]
        if _missing:
            raise AttributeError(f"BetaflightEnvCfg is missing required fields: {_missing}")

        # ---------------- Post-init buffers that depend on num_envs/device ----------------
        # Actions/controls
        act_dim = 4 * self._num_robots
        self._actions = torch.zeros(self.num_envs, act_dim, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)

        # Per-robot thrust/moment commands
        self._thrust = torch.zeros(self.num_envs, self._num_robots, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, self._num_robots, 3, device=self.device)

        # Desired goal position (world) and yaw (shared for group)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw = torch.zeros(self.num_envs, device=self.device)

        # For group: commanded angular vel and thrust per robot
        self._desired_ang_vel = torch.zeros(self.num_envs, self._num_robots, 3, device=self.device)
        self._commanded_ang_vel = torch.zeros(self.num_envs, self._num_robots, 3, device=self.device)
        self._commanded_thrust = torch.zeros(self.num_envs, self._num_robots, device=self.device)

        # Control system parameters from configuration
        self._max_ang_vel = math.pi * self.cfg.max_ang_vel_deg_s / 180.0  # rad/s
        self._ang_vel_tau = self.cfg.ang_vel_tau
        self._thrust_tau = self.cfg.thrust_tau

        # Goal sequencing / custom starts
        self._custom_goal = None
        self._custom_start = None
        self._goal_sequence = None
        self._goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Training/eval thrust constant
        self._thrust_constant = torch.full((self.num_envs, self._num_robots), self.cfg.eval_thrust_constant, device=self.device)
        self.is_training = self.cfg.is_training

        # Episode logging
        keys = [
            "tracking",
            "center_velocity",
            "neighbor_velocity",
            "ring_radial",
            "lin_vel",
            "ang_vel",
            "orientation",
            "thrust_smoothness",
            "roll_smoothness",
            "pitch_smoothness",
            "yaw_smoothness",
            "inter_drone_separation",
            "z_alignment",
            "collision_avoidance",
            "max_separation_violation",  # NEW
            "total_reward",
        ]
        self._episode_sums = {k: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for k in keys}

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    # --------------------------- Public setters ---------------------------
    def set_custom_goal(self, goal_tensor):
        self._custom_goal = goal_tensor.clone().to(self.device)

    def set_custom_start(self, start_tensor):
        """Center of formation (world frame)."""
        self._custom_start = start_tensor.clone().to(self.device)

    # ---------------------------- Scene setup ----------------------------
    def _setup_scene(self):
        # Terrain already configured via cfg; environments are cloned by base env
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Create robots here (so the scene initializes their views)
        if self.group_mode:
            self._robots = []
            for i in range(self._num_robots):
                cfg_i: ArticulationCfg = copy.deepcopy(self.cfg.robot)
                cfg_i.prim_path = f"/World/envs/env_.*/Robot{i}"
                robot_i = Articulation(cfg_i)
                self._robots.append(robot_i)
                self.scene.articulations[f"robot{i}"] = robot_i
        else:
            self._robot = Articulation(self.cfg.robot)
            self.scene.articulations["robot"] = self._robot

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------ Control application ------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        dt = self.cfg.sim.dt * self.cfg.decimation
        alpha_thrust = dt / (self._thrust_tau + dt)
        alpha_ang = dt / (self._ang_vel_tau + dt)

        if self.group_mode:
            # Split actions into N chunks of 4: [thrust, roll, pitch, yaw]
            for i in range(self._num_robots):
                r_actions = self._actions[:, 4*i:4*(i+1)]
                force = (r_actions[:, 0] + 1.0) / 2.0  # map [-1,1] -> [0,1]
                desired_thrust = self._thrust_constant[:, i] * force
                self._commanded_thrust[:, i] = (1.0 - alpha_thrust) * self._commanded_thrust[:, i] + alpha_thrust * desired_thrust

                desired_ang_vel = r_actions[:, 1:4] * self._max_ang_vel
                desired_ang_vel[:, 2] *= -1.0  # invert yaw
                self._desired_ang_vel[:, i] = desired_ang_vel

            # Current angular velocities per robot and PD torque
            curr_ang = [r.data.root_ang_vel_b for r in self._robots]  # each: [num_envs, 3]
            kp_rp = self.cfg.ang_vel_kp_roll_pitch
            kp_yaw = self.cfg.ang_vel_kp_yaw
            for i in range(self._num_robots):
                ang_err = self._desired_ang_vel[:, i] - curr_ang[i]
                self._commanded_ang_vel[:, i] = (1.0 - alpha_ang) * self._commanded_ang_vel[:, i] + alpha_ang * self._desired_ang_vel[:, i]
                self._moment[:, i, 0] = kp_rp * ang_err[:, 0]
                self._moment[:, i, 1] = kp_rp * ang_err[:, 1]
                self._moment[:, i, 2] = kp_yaw * ang_err[:, 2]

            # Apply thrust (body Z)
            self._thrust[:, :, 0] = 0.0
            self._thrust[:, :, 1] = 0.0
            self._thrust[:, :, 2] = self._commanded_thrust

        else:
            # Single robot path
            force = (self._actions[:, 0] + 1.0) / 2.0
            desired_thrust = self._thrust_constant[:, 0] * force
            self._commanded_thrust[:, 0] = (1.0 - alpha_thrust) * self._commanded_thrust[:, 0] + alpha_thrust * desired_thrust

            desired_ang_vel = self._actions[:, 1:4] * self._max_ang_vel
            desired_ang_vel[:, 2] *= -1.0
            self._desired_ang_vel[:, 0] = desired_ang_vel

            current_ang_vel = self._robot.data.root_ang_vel_b
            ang_err = self._desired_ang_vel[:, 0] - current_ang_vel
            self._commanded_ang_vel[:, 0] = (1.0 - alpha_ang) * self._commanded_ang_vel[:, 0] + alpha_ang * self._desired_ang_vel[:, 0]

            kp_rp = self.cfg.ang_vel_kp_roll_pitch
            kp_yaw = self.cfg.ang_vel_kp_yaw
            self._moment[:, 0, 0] = kp_rp * ang_err[:, 0]
            self._moment[:, 0, 1] = kp_rp * ang_err[:, 1]
            self._moment[:, 0, 2] = kp_yaw * ang_err[:, 2]

            self._thrust[:, 0, 0] = 0.0
            self._thrust[:, 0, 1] = 0.0
            self._thrust[:, 0, 2] = self._commanded_thrust[:, 0]

    def _apply_action(self) -> None:
        # set_external_force_and_torque expects per-body forces when body_ids=None.
        # Build [num_envs, num_bodies, 3] tensors and distribute thrust/torque across bodies.
        if self.group_mode:
            for i, robot in enumerate(self._robots):
                num_bodies = len(robot.data.body_names)
                forces = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
                torques = torch.zeros_like(forces)
                # Distribute thrust along +Z across all bodies so the net equals commanded thrust
                forces[:, :, 2] = self._commanded_thrust[:, i].unsqueeze(1) / max(num_bodies, 1)
                # Distribute body-frame moments similarly
                torques[:, :, 0] = self._moment[:, i, 0].unsqueeze(1) / max(num_bodies, 1)
                torques[:, :, 1] = self._moment[:, i, 1].unsqueeze(1) / max(num_bodies, 1)
                torques[:, :, 2] = self._moment[:, i, 2].unsqueeze(1) / max(num_bodies, 1)
                robot.set_external_force_and_torque(forces, torques, body_ids=None)
        else:
            num_bodies = len(self._robot.data.body_names)
            forces = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
            torques = torch.zeros_like(forces)
            forces[:, :, 2] = self._commanded_thrust[:, 0].unsqueeze(1) / max(num_bodies, 1)
            torques[:, :, 0] = self._moment[:, 0, 0].unsqueeze(1) / max(num_bodies, 1)
            torques[:, :, 1] = self._moment[:, 0, 1].unsqueeze(1) / max(num_bodies, 1)
            torques[:, :, 2] = self._moment[:, 0, 2].unsqueeze(1) / max(num_bodies, 1)
            self._robot.set_external_force_and_torque(forces, torques, body_ids=None)

    # --------------------------- Observations ----------------------------
    @staticmethod
    def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """quat: [..., 4] in [w, x, y, z] order -> yaw (rad)."""
        w, x, y, z = quat_wxyz.unbind(-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _get_observations(self) -> dict:
        if self.group_mode:
            # States per robot
            pos_w = [r.data.root_state_w[:, :3] for r in self._robots]
            quat_w = [r.data.root_state_w[:, 3:7] for r in self._robots]
            lin_vel_b = [r.data.root_lin_vel_b for r in self._robots]
            ang_vel_b = [r.data.root_ang_vel_b for r in self._robots]

            # Formation center and average yaw
            stack_pos = torch.stack(pos_w, dim=1)              # [N_env, N, 3]
            center_pos_w = stack_pos.mean(dim=1)               # [N_env, 3]
            yaws = [self._yaw_from_quat_wxyz(q) for q in quat_w]
            yaw_mat = torch.stack(yaws, dim=1)                 # [N_env, N]
            avg_yaw = torch.atan2(torch.sin(yaw_mat).mean(dim=1), torch.cos(yaw_mat).mean(dim=1))

            # Heading error wrt desired yaw
            yaw_err = torch.atan2(torch.sin(avg_yaw - self._desired_yaw), torch.cos(avg_yaw - self._desired_yaw))
            heading_error = torch.stack([torch.sin(yaw_err), torch.cos(yaw_err)], dim=-1)

            # Desired (goal - center) in world
            center_to_goal_w = self._desired_pos_w - center_pos_w  # [N_env,3]
            # Rotate center->goal vector from world into a common body-aligned frame (average yaw)
            dx, dy, dz = center_to_goal_w.unbind(-1)
            c = torch.cos(avg_yaw)
            s = torch.sin(avg_yaw)
            cx = c * dx + s * dy
            cy = -s * dx + c * dy
            center_to_goal_b = torch.stack([cx, cy, dz], dim=-1)
            # Observation: each robot's local states + center-goal vector + heading
                        
            # --- K-nearest neighbor relative vectors (per robot) ---
            K = getattr(self, "_neighbors_k_eff", max(0, int(getattr(self.cfg, "neighbors_k", 2))))# Pairwise relative vectors and distances in WORLD frame: [B, N, N, 3] and [B, N, N]
            rel_ij_w = stack_pos.unsqueeze(2) - stack_pos.unsqueeze(1)
            dist_ij = torch.linalg.norm(rel_ij_w, dim=-1) + torch.eye(self._num_robots, device=self.device)[None,...]*1e6
            # Indices of K nearest neighbors per robot (exclude self via large diag)
            
            if K > 0:
                # Provisional KNN by distance (exclude self via large diag)
                knn_idx = torch.topk(-dist_ij, k=K, dim=2).indices  # [B, N, K]

                # --- Stable ordering for neighbors to avoid tie flip-flops on symmetric rings ---
                # Sort selected neighbors by bearing in robot-i body frame (clockwise from +X_b)
                rows = torch.arange(stack_pos.shape[0], device=self.device)[:, None, None]
                cols = torch.arange(self._num_robots, device=self.device)[None, :, None]
                rel_knn_w_tmp = rel_ij_w[rows, cols, knn_idx]  # [B,N,K,3]
                dxw = rel_knn_w_tmp[..., 0]; dyw = rel_knn_w_tmp[..., 1]
                c_i = torch.cos(yaw_mat)[:, :, None]; s_i = torch.sin(yaw_mat)[:, :, None]
                dxb =  c_i * dxw + s_i * dyw
                dyb = -s_i * dxw + c_i * dyw
                ang = torch.atan2(dyb, dxb)  # [-pi, pi]
                order = torch.argsort(ang, dim=-1)  # ascending angle

                batch = torch.arange(knn_idx.shape[0], device=self.device)[:, None, None]
                robot = torch.arange(knn_idx.shape[1], device=self.device)[None, :, None]
                knn_idx = knn_idx[batch, robot, order]  # [B,N,K] (deterministic)
            else:
                # No neighbors requested
                knn_idx = torch.empty((stack_pos.shape[0], self._num_robots, 0), dtype=torch.long, device=self.device)  # [B,N,0]
            # Yaw of each robot: [B, N]
            # Build per-robot rotation to body frame for its neighbors
            c_i = torch.cos(yaw_mat)[:, :, None]  # [B,N,1]
            s_i = torch.sin(yaw_mat)[:, :, None]
            # Gather relative vectors to K neighbors: [B, N, K, 3]
            rows = torch.arange(stack_pos.shape[0], device=self.device)[:, None, None]
            cols = torch.arange(self._num_robots, device=self.device)[None, :, None]
            rel_knn_w = rel_ij_w[rows, cols, knn_idx] if K>0 else torch.zeros((stack_pos.shape[0], self._num_robots, 0, 3), device=self.device)  # [B,N,K,3]
            dx = rel_knn_w[..., 0]; dy = rel_knn_w[..., 1]; dz = rel_knn_w[..., 2]
            # Rotate into robot-i body frame (yaw only)
            dx_b = c_i * dx + s_i * dy
            dy_b = -s_i * dx + c_i * dy
            rel_knn_b = torch.stack([dx_b, dy_b, dz], dim=-1) if K>0 else rel_knn_w  # [B,N,K,3]
            # Flatten per robot
            rel_knn_flat = rel_knn_b.reshape(rel_knn_b.shape[0], rel_knn_b.shape[1], K*3)

            obs_chunks = []
            for i in range(self._num_robots):
                obs_chunks.extend([lin_vel_b[i], ang_vel_b[i], quat_w[i], self._prev_actions[:, 4*i:4*(i+1)], rel_knn_flat[:, i, :]])
            obs = torch.cat([*obs_chunks, center_to_goal_b, heading_error], dim=-1)
            return {"policy": obs}

        else:
            quad_pos_w = self._robot.data.root_state_w[:, :3]
            quad_quat_w = self._robot.data.root_state_w[:, 3:7]

            desired_pos_b, _ = subtract_frame_transforms(
                quad_pos_w, quad_quat_w, self._desired_pos_w
            )
            curr_yaw = self._yaw_from_quat_wxyz(quad_quat_w)
            yaw_err = torch.atan2(torch.sin(curr_yaw - self._desired_yaw), torch.cos(curr_yaw - self._desired_yaw))
            heading_error = torch.stack([torch.sin(yaw_err), torch.cos(yaw_err)], dim=-1)

            obs = torch.cat([
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                quad_quat_w,
                desired_pos_b,
                heading_error,
                self._prev_actions[:, 0:4],
            ], dim=-1)
            return {"policy": obs}

    # ----------------------------- Rewards -------------------------------
    def _get_rewards(self) -> torch.Tensor:
        if self.group_mode:
            pos_w = [r.data.root_state_w[:, :3] for r in self._robots]
            lin_vel_b = [r.data.root_lin_vel_b for r in self._robots]
            ang_vel_b = [r.data.root_ang_vel_b for r in self._robots]

            # Formation center tracking to goal
            stack_pos = torch.stack(pos_w, dim=1)             # [N_env, N, 3]
            center_pos_w = stack_pos.mean(dim=1)              # [N_env, 3]
            distance_to_goal = torch.linalg.norm(self._desired_pos_w - center_pos_w, dim=1)
            distance_reward = 1.0 - torch.tanh(distance_to_goal / self.cfg.distance_normalizer)
            tracking = distance_reward * self.cfg.distance_to_goal_reward_scale * self.step_dt

            # Smoothness penalties (per-robot)
            action_diff = self._actions - self._prev_actions
            # Every 4th action is thrust, then roll, pitch, yaw across robots
            thrust_rate = action_diff[:, 0::4].pow(2).sum(dim=1)
            roll_rate   = action_diff[:, 1::4].pow(2).sum(dim=1)
            pitch_rate  = action_diff[:, 2::4].pow(2).sum(dim=1)
            yaw_rate    = action_diff[:, 3::4].pow(2).sum(dim=1)

            thrust_smooth_penalty = thrust_rate * self.cfg.thrust_smoothness_penalty_scale * self.step_dt
            roll_smooth_penalty   = roll_rate   * self.cfg.roll_smoothness_penalty_scale   * self.step_dt
            pitch_smooth_penalty  = pitch_rate  * self.cfg.pitch_smoothness_penalty_scale  * self.step_dt
            yaw_smooth_penalty    = yaw_rate    * self.cfg.yaw_smoothness_penalty_scale    * self.step_dt

            # Velocities/angles penalties (mean across robots)
            lin_vel = torch.stack([v.pow(2).sum(dim=1) for v in lin_vel_b], dim=1).mean(dim=1)
            ang_vel = torch.stack([w.pow(2).sum(dim=1) for w in ang_vel_b], dim=1).mean(dim=1)
            lin_vel_term = lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
            ang_vel_term = ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt

            # Heading penalty based on average yaw error
            quat_w = [r.data.root_state_w[:, 3:7] for r in self._robots]
            yaws = [self._yaw_from_quat_wxyz(q) for q in quat_w]
            yaw_mat = torch.stack(yaws, dim=1)  # [N_env, N]
            avg_yaw = torch.atan2(torch.sin(yaw_mat).mean(dim=1), torch.cos(yaw_mat).mean(dim=1))
            yaw_err = torch.atan2(torch.sin(avg_yaw - self._desired_yaw), torch.cos(avg_yaw - self._desired_yaw))
            heading_pen = (1 - torch.cos(yaw_err)) * self.cfg.orientation_penalty_scale * self.step_dt

            # Keep drones at the same altitude (penalize variance of z across robots)
            z_vals = stack_pos[:, :, 2]  # [N_env, N]
            z_mean = z_vals.mean(dim=1, keepdim=True)
            z_var = (z_vals - z_mean).pow(2).mean(dim=1)  # variance across robots
            z_align_pen = z_var * self.cfg.z_alignment_penalty_scale * self.step_dt

            # --- Formation radial penalty (encourage each drone to stay on target ring radius) ---
            N = self._num_robots
            side = self.cfg.formation_target_distance
            target_radius = side / (2.0 * math.sin(math.pi / max(N, 2)))
            rel_from_center = stack_pos - center_pos_w.unsqueeze(1)              # [B,N,3]
            radii = torch.linalg.norm(rel_from_center, dim=2)                     # [B,N]
            radial_err = (radii - target_radius).pow(2).mean(dim=1)
            ring_radial_pen = radial_err * self.cfg.ring_radial_penalty_scale * self.step_dt

            # --- Velocity consensus & center drift damping ---
            stack_vel_w = torch.stack([r.data.root_state_w[:, 7:10] for r in self._robots], dim=1)  # [B,N,3]
            rolled_vel = torch.roll(stack_vel_w, shifts=-1, dims=1)
            edge_vel_diff = torch.linalg.norm(stack_vel_w - rolled_vel, dim=2).pow(2).mean(dim=1)   # [B]
            neighbor_vel_pen = edge_vel_diff * self.cfg.neighbor_velocity_penalty_scale * self.step_dt

            center_vel = stack_vel_w.mean(dim=1)                         # [B,3]
            center_vel_pen = center_vel.pow(2).sum(dim=1) * self.cfg.center_velocity_penalty_scale * self.step_dt

            # Inter-drone separation: encourage ring with side length = formation_target_distance
            # Adjacent pairs on the ring
            rolled = torch.roll(stack_pos, shifts=-1, dims=1)               # neighbor i+1
            edge_dists = torch.linalg.norm(stack_pos - rolled, dim=2)       # [N_env, N]
            sep_error = edge_dists - self.cfg.formation_target_distance
            separation_pen = (sep_error.pow(2).mean(dim=1)) * self.cfg.formation_separation_penalty_scale * self.step_dt

            # Strong collision avoidance for ALL pairs
            # pairwise distances [N_env, N, N]
            pairwise = torch.cdist(stack_pos, stack_pos, p=2)
            # Only count i<j
            mask = torch.triu(torch.ones((self._num_robots, self._num_robots), device=self.device), diagonal=1).bool()
            close = (self.cfg.min_separation - pairwise).clamp(min=0.0).pow(2)
            collision_pen = close[:, mask].sum(dim=1) * self.cfg.collision_penalty_scale * self.step_dt

            # --- NEW: Extreme penalty when any pair exceeds max separation ---
            far_excess = (pairwise - self.cfg.max_pairwise_separation).clamp(min=0.0)
            far_violation = far_excess[:, mask]  # [B, num_edges]
            # Sum of squared violations across all edges (only when exceeded)
            far_cost = far_violation.pow(2).sum(dim=1)
            max_sep_pen = far_cost * self.cfg.max_separation_penalty_scale * self.step_dt

            rewards = {
                "tracking": tracking,
                "ring_radial": ring_radial_pen,
                "neighbor_velocity": neighbor_vel_pen,
                "center_velocity": center_vel_pen,
                "lin_vel": lin_vel_term,
                "ang_vel": ang_vel_term,
                "orientation": heading_pen,
                "thrust_smoothness": thrust_smooth_penalty,
                "roll_smoothness": roll_smooth_penalty,
                "pitch_smoothness": pitch_smooth_penalty,
                "yaw_smoothness": yaw_smooth_penalty,
                "inter_drone_separation": separation_pen,
                "collision_avoidance": collision_pen,
                "z_alignment": z_align_pen,
                "max_separation_violation": max_sep_pen,  # NEW
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            for k, v in rewards.items():
                if k not in self._episode_sums:
                    self._episode_sums[k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                self._episode_sums[k] += v
            self._episode_sums["total_reward"] += reward
            return reward

        else:
            # Original single-drone rewards
            distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
            distance_reward = 1.0 - torch.tanh(distance_to_goal / self.cfg.distance_normalizer)
            distance_scale = self.cfg.distance_to_goal_reward_scale

            lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

            q = self._robot.data.root_state_w[:, 3:7]  # [w,x,y,z]
            curr_yaw = self._yaw_from_quat_wxyz(q)
            yaw_err = torch.atan2(torch.sin(curr_yaw - self._desired_yaw), torch.cos(curr_yaw - self._desired_yaw))
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
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            for k, v in rewards.items():
                if k not in self._episode_sums:
                    self._episode_sums[k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                self._episode_sums[k] += v
            self._episode_sums["total_reward"] += reward
            return reward

    # ------------------------------ Dones --------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.group_mode:
            # Any robot outside Z-bounds ends the episode
            zs = [r.data.root_pos_w[:, 2] for r in self._robots]
            below = torch.stack([z < 0.1 for z in zs], dim=1).any(dim=1)
            above = torch.stack([z > 3.0 for z in zs], dim=1).any(dim=1)
            died = below | above

            # --- NEW: optional termination when any pair separates beyond threshold ---
            if getattr(self.cfg, "terminate_on_max_separation_exceeded", False):
                pos_w = [r.data.root_state_w[:, :3] for r in self._robots]
                stack_pos = torch.stack(pos_w, dim=1)  # [N_env, N, 3]
                pairwise = torch.cdist(stack_pos, stack_pos, p=2)  # [N_env, N, N]
                max_pair = torch.amax(pairwise, dim=(1, 2))
                died = died | (max_pair > self.cfg.max_pairwise_separation)

        else:
            died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 3.0)
        return died, time_out

    # ------------------------------ Reset --------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            if self.group_mode:
                env_ids = self._robots[0]._ALL_INDICES
            else:
                env_ids = self._robot._ALL_INDICES

        # Reset robot handles
        if self.group_mode:
            for r in self._robots:
                r.reset(env_ids)
        else:
            self._robot.reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

        self._actions[env_ids] = 0.0

        # Thrust constant (per-robot when in group)
        if self.cfg.thrust_constant_train_only and self.is_training:
            mean = self.cfg.thrust_constant_gauss_mean
            std = self.cfg.thrust_constant_gauss_std
            for i in range(self._num_robots):
                samples = torch.normal(mean=torch.full((len(env_ids),), mean, device=self.device),
                                       std=torch.full((len(env_ids),), std, device=self.device))
                samples = samples.clamp(self.cfg.thrust_constant_clip_min, self.cfg.thrust_constant_clip_max)
                self._thrust_constant[env_ids, i] = samples
        else:
            self._thrust_constant[env_ids] = self.cfg.eval_thrust_constant

        self._desired_ang_vel[env_ids] = 0.0
        self._commanded_ang_vel[env_ids] = 0.0
        self._commanded_thrust[env_ids] = 0.0
        self._desired_yaw[env_ids] = 0.0

        # Sample new commands / goal
        if self._custom_goal is not None:
            self._desired_pos_w[env_ids] = self._custom_goal.expand(len(env_ids), -1)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        elif self._goal_sequence is not None:
            self._goal_idx[env_ids] = 0
            self._desired_pos_w[env_ids] = self._goal_sequence[self._goal_idx[env_ids]].expand(len(env_ids), -1)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        else:
            self._desired_pos_w[env_ids, :2] = (
                torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
                + self._terrain.env_origins[env_ids, :2]
            )
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 2.0)

        # Reset robot(s) state(s)
        if self.group_mode:
            # Formation center (base) for these envs
            if self._custom_start is not None:
                base_center = self._custom_start.expand(len(env_ids), -1)
            else:
                base_center = torch.zeros((len(env_ids), 3), device=self.device)
            base_center = base_center + self._terrain.env_origins[env_ids]

            # Regular N-gon with side length = formation_target_distance at z=0.5
            N = self._num_robots
            if N == 1:
                radii = torch.zeros((N,), device=self.device)
                angles = torch.zeros((N,), device=self.device)
            else:
                side = self.cfg.formation_target_distance
                radius = side / (2.0 * math.sin(math.pi / N))
                angles = torch.linspace(0, 2*math.pi, steps=N+1, device=self.device)[:-1]  # [0, 2pi)
                radii = torch.full((N,), radius, device=self.device)

            # For each robot i, build default root state and write to sim
            for i in range(N):
                default_root_state = self._robots[i].data.default_root_state[env_ids].clone()
                # XY offset on circle
                dx = radii[i] * torch.cos(angles[i])
                dy = radii[i] * torch.sin(angles[i])
                offset = torch.stack([dx.expand(len(env_ids)), dy.expand(len(env_ids)), torch.full((len(env_ids),), 0.5, device=self.device)], dim=1)
                default_root_state[:, :3] = base_center + offset
                self._robots[i].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
                self._robots[i].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
                # Reset joints if any
                jp = self._robots[i].data.default_joint_pos[env_ids]
                jv = self._robots[i].data.default_joint_vel[env_ids]
                self._robots[i].write_joint_state_to_sim(jp, jv, None, env_ids)
        else:
            joint_pos = self._robot.data.default_joint_pos[env_ids]
            joint_vel = self._robot.data.default_joint_vel[env_ids]
            default_root_state = self._robot.data.default_root_state[env_ids]
            if self._custom_start is not None:
                default_root_state[:, :3] = self._custom_start.expand(len(env_ids), -1)
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ---------------------- Debug visualization --------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = copy.deepcopy(CUBOID_MARKER_CFG)
                marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    # ---------------------------- Utilities ------------------------------
    def set_goal_sequence(self, goal_list: list[torch.Tensor]):
        self._goal_sequence = [g.to(self.device) for g in goal_list]
        self._goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._desired_pos_w = torch.stack([g for g in goal_list[:self.num_envs]]).to(self.device)
