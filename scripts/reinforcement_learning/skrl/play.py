# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument( "--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.", )
parser.add_argument("--is_training", type=bool, default=True, help="Train or inference.")
parser.add_argument( "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--goal_pos", type=float, nargs=3, help="Custom goal position (x y z)")
parser.add_argument("--start_pos", type=float, nargs=3, help="Custom starting position (x y z)")
parser.add_argument("--goal_sequence", action="store_true", default=False, help="Use a sequence of goals.")
parser.add_argument("--log_data", action="store_true", default=False, help="Log data during play.")

# append AppLauncher cli args
parser.add_argument("--pure_pursuit", action="store_true", default=True, help="Enable pure-pursuit circular trajectory")
parser.add_argument("--pp_lookahead_m", type=float, default=0.5, help="Pure pursuit lookahead (arc length, meters)")
parser.add_argument("--pp_direction", type=str, default="cw", choices=["cw", "ccw"], help="Circle direction for pure pursuit")
parser.add_argument("--pp_max_step_deg", type=float, default=45.0, help="Clamp per-tick setpoint jump (deg of arc)")

# Circle/plane/periodic parameters (mirroring main.py)
parser.add_argument("--goal_shape", type=str, default="circle", help="Path shape: 'circle' or 'square' (PP only supports 'circle')")
parser.add_argument("--goal_circle_radius", type=float, default=1.2, help="Circle radius (m)")
parser.add_argument("--goal_center_xy", type=float, nargs=2, default=[0.0, 0.0], help="Circle center [cx cy] in world (m)")
parser.add_argument("--goal_altitude", type=float, default=1.91, help="Base altitude z (m)")
parser.add_argument("--goal_z_plane_gx", type=float, default=0.0, help="Plane tilt dz/dx")
parser.add_argument("--goal_z_plane_gy", type=float, default=0.0, help="Plane tilt dz/dy")
parser.add_argument("--goal_circle_dz_per_rev", type=float, default=0.0, help="Helix vertical rise per revolution (m/2π)")
parser.add_argument("--goal_circle_z_sin", type=float, default=0.0, help="Periodic z term: +a*sin(theta)")
parser.add_argument("--goal_circle_z_cos", type=float, default=0.0, help="Periodic z term: +b*cos(theta)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import math
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    task_name = args_cli.task.split(":")[-1]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    if args_cli.log_data:
        payload_log = []

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # --- Pure Pursuit: init state & circle params ---
    _pp_enabled = bool(args_cli.pure_pursuit) and (str(args_cli.goal_shape).lower() == "circle")
    _pp_last_theta = None
    _pp_theta_unwrapped = 0.0

    _pp_cx, _pp_cy = float(args_cli.goal_center_xy[0]), float(args_cli.goal_center_xy[1])
    _pp_r = float(args_cli.goal_circle_radius)
    _pp_base_z = float(args_cli.goal_altitude)
    _pp_gx = float(args_cli.goal_z_plane_gx)
    _pp_gy = float(args_cli.goal_z_plane_gy)
    _pp_a_sin = float(args_cli.goal_circle_z_sin)
    _pp_a_cos = float(args_cli.goal_circle_z_cos)
    _pp_dz_rev = float(args_cli.goal_circle_dz_per_rev)

    # helper to update desired goal per tick for all envs
    def _pp_update_goal():
        nonlocal _pp_last_theta, _pp_theta_unwrapped
        if not _pp_enabled or args_cli.goal_pos is not None or args_cli.goal_sequence:
            return
        try:
            # --- Get reference position (env 0) ---
            # If group mode, use formation center; else single robot root pose
            if hasattr(env.unwrapped, "_robots"):
                # Group mode: average of all robots' world positions (env 0)
                pos_w_list = [r.data.root_state_w for r in env.unwrapped._robots]  # each [num_envs, 13]
                stack_pos = torch.stack([p[:, :3] for p in pos_w_list], dim=1)      # [num_envs, N, 3]
                center_pos = stack_pos.mean(dim=1)                                  # [num_envs, 3]
                px = float(center_pos[0, 0].item())
                py = float(center_pos[0, 1].item())
            else:
                # Single robot
                root = env.unwrapped._robot.data.root_state_w  # (num_envs, 13)
                px = float(root[0, 0].item())
                py = float(root[0, 1].item())
        except Exception:
            return

        # compute current angle around circle center
        theta_now = math.atan2(py - _pp_cy, px - _pp_cx)

        # unwrap theta to be continuous
        if _pp_last_theta is None:
            _pp_last_theta = theta_now
            _pp_theta_unwrapped = theta_now
        else:
            dtheta = math.atan2(math.sin(theta_now - _pp_last_theta), math.cos(theta_now - _pp_last_theta))
            _pp_theta_unwrapped += dtheta
            _pp_last_theta = theta_now

        # arc-length -> angle increment; direction
        lookahead_m = float(args_cli.pp_lookahead_m)
        sgn = -1.0 if str(args_cli.pp_direction).lower().strip() == "cw" else 1.0
        dtheta_la = sgn * (lookahead_m / max(_pp_r, 1e-6))

        # clamp per tick
        max_step = math.radians(float(args_cli.pp_max_step_deg))
        if dtheta_la >  max_step: dtheta_la =  max_step
        if dtheta_la < -max_step: dtheta_la = -max_step

        theta_la_unwrapped = _pp_theta_unwrapped + dtheta_la
        # new XY setpoint
        gx = _pp_cx + _pp_r * math.cos(theta_la_unwrapped)
        gy = _pp_cy + _pp_r * math.sin(theta_la_unwrapped)

        # Z = plane + periodic + helix
        plane_z = _pp_base_z + _pp_gx * (gx - _pp_cx) + _pp_gy * (gy - _pp_cy)
        z_helix = (_pp_dz_rev / (2.0 * math.pi)) * theta_la_unwrapped
        # avoid redundant atan2-sin/cos calls; just reuse theta
        z_periodic = _pp_a_sin * math.sin(theta_la_unwrapped) + _pp_a_cos * math.cos(theta_la_unwrapped)
        gz = plane_z + z_periodic + z_helix

        # Build per-env goals adding each env's XY origin, z unchanged
        try:
            origins = env.unwrapped._terrain.env_origins  # (num_envs, 3) torch tensor
            num = origins.shape[0]
            goal = torch.zeros((num, 3), dtype=origins.dtype, device=origins.device)
            goal[:, 0] = gx + origins[:, 0]
            goal[:, 1] = gy + origins[:, 1]
            goal[:, 2] = gz
            env.unwrapped._desired_pos_w = goal
        except Exception:
            # fallback: set a single goal for all envs (no origin offset)
            env.unwrapped.set_custom_goal(torch.tensor([gx, gy, gz], dtype=torch.float32))


    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Set goal/start positions
    if args_cli.goal_pos:
        goal_tensor = torch.tensor(args_cli.goal_pos, dtype=torch.float32)
        env.unwrapped.set_custom_goal(goal_tensor)
        print(f"[INFO] Using custom GOAL position: {args_cli.goal_pos}")
    if args_cli.start_pos:
        start_tensor = torch.tensor(args_cli.start_pos, dtype=torch.float32)
        env.unwrapped.set_custom_start(start_tensor)
        print(f"[INFO] Using custom START position: {args_cli.start_pos}")

    if args_cli.goal_sequence:
        goal_sequence = [
            # torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            # torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
            torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32),
            torch.tensor([1.0, 3.0, 1.0], dtype=torch.float32),
            torch.tensor([0.0, 4.0, 1.0], dtype=torch.float32),
            torch.tensor([1.0, 3.0, 1.0], dtype=torch.float32),
            torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        ]
        # goal_sequence = generate_trajectory(
        #     path_type="sawtooth",  # "circle", "sine", "sawtooth"
        #     num_points=4,  # number of waypoints in the trajectory
        #     radius=1.0,  # radius for circle or line length
        #     center=(0.0, 0.0, 2),  # center point for circle or line start point
        #     amplitude=2,  # amplitude for sine wave
        #     frequency=7,  # frequency for sine wave
        # )
        env.unwrapped.set_goal_sequence(goal_sequence)
        print(f"[INFO] Using FIXED GOAL SEQUENCE with {len(goal_sequence)} waypoints")

    if args_cli.is_training is False:
        env.unwrapped.is_training = False
        

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    printLimit = 150
    # set print options for torch
    torch.set_printoptions(precision=3, sci_mode=False)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # Update PP setpoint (before inference/step so next obs sees new goal)
        try:
            _pp_update_goal()
        except Exception:
            pass


        if timestep < printLimit:
            print(f"Timestep {timestep}: Obs: {obs.cpu().numpy().round(3)}")
            if args_cli.log_data:
                payload_log.append(obs[0].tolist())

        # run everything in inference mode
        with torch.inference_mode():
            # round obs before passing to policy
            obs = torch.round(obs, decimals=3)
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

        # print observation
        if timestep < printLimit:
            print(f"Action: {actions.detach().cpu().numpy().round(3)}")
            if args_cli.log_data:
                payload_log.append([actions[0][2].item() if hasattr(actions, "item") else actions[0][2]])
        

        timestep += 1
        # exit the play loop after 200 steps
        if timestep >= 4000:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.log_data and payload_log:
        import csv

        payload_log_path = os.path.join(log_dir, "obs_act_log.csv")

        with open(payload_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(payload_log)
        print(f"[INFO] Log saved to: {payload_log_path}")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
