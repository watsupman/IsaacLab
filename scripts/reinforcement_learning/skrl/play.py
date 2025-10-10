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
    # set print options for torch
    torch.set_printoptions(precision=3, sci_mode=False)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        if timestep < 20:
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
        if timestep < 20:
            print(f"Action: {actions.detach().cpu().numpy().round(3)}")
            if args_cli.log_data:
                payload_log.append([actions[0][2].item() if hasattr(actions, "item") else actions[0][2]])
        

        timestep += 1
        # exit the play loop after 200 steps
        if timestep >= 400:
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
