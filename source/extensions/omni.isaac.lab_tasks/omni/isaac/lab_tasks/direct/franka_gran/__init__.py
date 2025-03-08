# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .franka_gran_cfg import FrankaGranCfg
from .franka_gran_env import FrankaGranIK
from .franka_gran_yaw import FrankaGranYaw
from .franka_gran_spawn_close import FrankaGranSpawnClose
from .franka_gran_spawn_close2 import FrankaGranSpawnClose2
from .franka_gran_spawn_close_2D_3dof import FrankaGranSpawnClose3DOF

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Granular-IK-v0",
    entry_point=f"{__name__}.franka_gran_env:FrankaGranIK",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_cfg:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Granular-Yaw-v0",
    entry_point=f"{__name__}.franka_gran_yaw:FrankaGranyaw",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_cfg:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Granular-Spawn-Close-v0",
    entry_point=f"{__name__}.franka_gran_spawn_close:FrankaGranSpawnClose",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Granular-Spawn-Close-v1",
    entry_point=f"{__name__}.franka_gran_spawn_close_2D_3dof:FrankaGranSpawnClose3DOF",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Granular-Spawn-Close-v2",
    entry_point=f"{__name__}.franka_gran_spawn_close2:FrankaGranSpawnClose2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
