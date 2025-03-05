# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .franka_EE_env_cfg import FrankaEndEffectorEnvCfg
from .franka_EE_env import FrankaEndEffectorEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-End-Effector-Direct-v0",
    entry_point=f"{__name__}.franka_EE_env:FrankaEndEffectorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_EE_env_cfg:FrankaEndEffectorEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
