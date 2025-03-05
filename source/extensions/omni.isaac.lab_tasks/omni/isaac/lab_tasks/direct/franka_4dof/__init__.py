# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .franka_4dof_cfg import Franka4dofCfg
from .franka_4dof import Franka4dof

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-4dof-Direct-ppo-v0",
    entry_point=f"{__name__}.franka_4dof:Franka4dof",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_4dof_cfg:Franka4dofCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPushObjectsPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-4dof-Direct-sac-v0",
    entry_point=f"{__name__}.franka_4dof:Franka4dof",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_4dof_cfg:Franka4dofCfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)
