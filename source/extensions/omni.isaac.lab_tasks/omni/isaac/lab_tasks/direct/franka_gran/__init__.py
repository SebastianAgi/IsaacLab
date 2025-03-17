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
from .franka_gran_2D_3DOF import FrankaGran2D3DOF
from .franka_gran_2D_3DOF_attract import FrankaGran2D3DOFAttract
from .franka_gran_2D_3DOF_attract_targetorigin import FrankaGran2D3DOFAttractTargetOrigin
from .franka_gran_2D_3DOF_new_reset import FrankaGran2D3DOFNewReset
from .franka_gran_2D_3DOF_reset_2 import FrankaGran2D3DOFReset2
from .franka_gran_2D_3DOF_grid import FrankaGran2D3DOFGrid

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


# Env config with action space of 2D and 3DOF movement ony on x,y axis
# Rewards here are objects distance to target and penalty for wrong direction
gym.register(
    id="Isaac-Franka-Granular-2D-3DOF-v0",
    entry_point=f"{__name__}.franka_gran_2D_3DOF:FrankaGran2D3DOF",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# Env config with action space of 2D and 3DOF movement ony on x,y axis
# Rewards here:
#    - objects distance to target
#    - Penalty for wrong direction
#    - Gripper attraction to objects
gym.register(
    id="Isaac-Franka-Granular-2D-3DOF-v1",
    entry_point=f"{__name__}.franka_gran_2D_3DOF_attract:FrankaGran2D3DOFAttract",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# Env config with action space of 2D and 3DOF movement ony on x,y axis, 
# and target is set to observation origin (not tested yet)
# Rewards here: 
#    - objects distance to target
#    - Penalty for wrong direction
#    - Gripper attraction to objects
gym.register(
    id="Isaac-Franka-Granular-2D-3DOF-v2",
    entry_point=f"{__name__}.franka_gran_2D_3DOF_attract_targetorigin:FrankaGran2D3DOFAttractTargetOrigin",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# Env config:
#    - action space of 2D and 3DOF movement ony on x,y axis, 
#    - target is set to observation origin
#    - New reset method
#    - Set to spawn close to objects and encourage task completion for reset experience
# Rewards here: 
#    - objects distance to target
#    - Penalty for wrong direction
#    - Gripper attraction to objects
gym.register(
    id="Isaac-Franka-Granular-2D-3DOF-v3",
    entry_point=f"{__name__}.franka_gran_2D_3DOF_new_reset:FrankaGran2D3DOFNewReset",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# Env config:
#    - action space of 2D and 3DOF movement ony on x,y axis, 
#    - target is set to observation origin
#    - New reset method
#    - Set to spawn close to objects and encourage task completion for reset experience
# Rewards here: 
#    - objects distance to target
#    - Penalty for wrong direction
#    - Gripper attraction to objects
gym.register(
    id="Isaac-Franka-Granular-2D-3DOF-v4",
    entry_point=f"{__name__}.franka_gran_2D_3DOF_reset_2:FrankaGran2D3DOFReset2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# Env config:
#    - action space of 2D and 3DOF movement ony on x,y axis, 
#    - target is set to observation origin
#    - New reset method
#    - Set to spawn close to objects and encourage task completion for reset experience
    #  - Observations is a 64x64 matrix of the table
# Rewards here: 
#    - objects distance to target
#    - Gripper attraction to objects
gym.register(
    id="Isaac-Franka-Granular-2D-3DOF-v5",
    entry_point=f"{__name__}.franka_gran_2D_3DOF_grid:FrankaGran2D3DOFGrid",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_gran_spawn_close:FrankaGranCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)




# Env config for placing gripper close to objects and also doing an initial action forcing of pushing 
# from the spawn location to target (did not work well)
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
