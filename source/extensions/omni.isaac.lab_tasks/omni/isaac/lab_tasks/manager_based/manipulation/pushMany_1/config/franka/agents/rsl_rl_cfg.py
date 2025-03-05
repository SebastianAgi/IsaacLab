# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class pushManyCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 50
    experiment_name = "franka_push_many_1"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, # Try to see what happens when we change this value
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2, # Try Sweep 0.1, 0.2, 0.3 0.4 0.5
        entropy_coef=0.006, # Try Sweep 0.01, 0.005, 0.001 0.0
        num_learning_epochs=5,
        num_mini_batches=4, # try 32
        learning_rate=5.0e-5, # rsl_rl learning rate: 1.0e-4 | rl_games learning rate: 5e-4
        schedule="adaptive", # Try 'None'
        gamma=0.98, # Try Sweep 0.99 0.995
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
