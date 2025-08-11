# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import math

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
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

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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
            torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
            # torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
            torch.tensor([0.0, 2.0, 2.0], dtype=torch.float32),
            torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32),
            torch.tensor([0.0, 4.0, 2.0], dtype=torch.float32),
            torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32),
            torch.tensor([0.0, 2.0, 2.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
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

    if args_cli.log_data:
        payload_log = []
        num_envs = args_cli.num_envs
        episode_counts = [0 for _ in range(num_envs)]

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

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
    obsCount = 0
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            # obs, _, _, _, _ = env.step(actions)
            obs, _, terminated, truncated, _ = env.step(actions)
            if obsCount < 4:
                obsCount += 1
                print(f"Obs {obsCount}: {obs[0].cpu().numpy().round(3)}")
                print(f"Actions: {actions[0].cpu().numpy().round(3)}")
            if args_cli.log_data:
                dones = torch.logical_or(terminated, truncated)
                relative_payload_pos ,relative_payload_vel = env.unwrapped.get_payload_state()
                distance_to_goal = torch.norm(
                    env.unwrapped._desired_pos_w - env.unwrapped._robot.data.root_pos_w, dim=1
                )
                for i in range(num_envs):
                    # Mark episode transition
                    if dones[i]:
                        episode_counts[i] += 1

                    payload_log.append({
                        "timestep": timestep,
                        "env_id": i,
                        "episode": episode_counts[i],
                        "rel_pos_x": relative_payload_pos[i][0].cpu().item(),
                        "rel_pos_y": relative_payload_pos[i][1].cpu().item(),
                        "rel_pos_z": relative_payload_pos[i][2].cpu().item(),
                        "rel_vel_x": relative_payload_vel[i][0].cpu().item(),
                        "rel_vel_y": relative_payload_vel[i][1].cpu().item(),
                        "rel_vel_z": relative_payload_vel[i][2].cpu().item(),
                        "dist_to_goal": distance_to_goal[i].cpu().item(),
                    })
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.log_data and payload_log:
        # save payload log to a file
        import pandas as pd

        payload_df = pd.DataFrame(payload_log)
        payload_log_path = os.path.join(log_dir, "payload_log.csv")
        payload_df.to_csv(payload_log_path, index=False)
        print(f"[INFO] Payload log saved to: {payload_log_path}")

    # close the simulator
    env.close()


def generate_trajectory(path_type="circle", num_points=100, radius=1.0, center=(0.0, 0.0, 2), amplitude=0.5, frequency=1.0):
    points = []

    if path_type == "circle":
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2]
            points.append(torch.tensor([x, y, z], dtype=torch.float32))

    elif path_type == "sine":
        for i in range(num_points):
            x = i * 2 * math.pi / num_points
            y = amplitude * math.sin(frequency * x)
            z = center[2]
            points.append(torch.tensor([x + center[0], y + center[1], z], dtype=torch.float32))

    elif path_type == "line":
        for i in range(num_points):
            alpha = i / (num_points - 1)
            x = center[0] + alpha * radius
            y = center[1]
            z = center[2]
            points.append(torch.tensor([x, y, z], dtype=torch.float32))
    elif path_type == "sawtooth":
        for i in range(num_points):
            x = center[0] + (amplitude if i % 2 == 0 else -amplitude)
            y = center[1] + i  # Move up by 1m per step
            z = center[2]
            points.append(torch.tensor([x, y, z], dtype=torch.float32))

    return points


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
