# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.g1"


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


for task_id, env_cfg in (
    ("Instinct-Parkour-Target-Amp-G1-23dof-v0", "G1Parkour23DofEnvCfg"),
    ("Instinct-Parkour-Target-Amp-G1-23dof-Play-v0", "G1Parkour23DofEnvCfg_PLAY"),
):
    gym.register(
        id=task_id,
        entry_point="instinctlab.envs:InstinctRlEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{task_entry}.g1_23dof_parkour_target_amp_cfg:{env_cfg}",
            "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1Parkour23DofPPORunnerCfg",
        },
    )
