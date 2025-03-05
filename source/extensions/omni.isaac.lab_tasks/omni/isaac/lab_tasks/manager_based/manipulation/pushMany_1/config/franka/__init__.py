# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents,joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Push-ManyCube-Franka-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaManyPushEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.pushManyCubePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)



