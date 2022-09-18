#!/usr/bin/env python3

from .envs.localize_env import LocalizeGibsonEnv


def create_env(params, pfnet_model):
    env = LocalizeGibsonEnv(
        config_file=params.config_file,
        scene_id=params.scene_id,
        mode=params.env_mode,
        action_timestep=params.action_timestep,
        physics_timestep=params.physics_timestep,
        device_idx=params.device_idx,
        pfnet_model=pfnet_model,
        pf_params=params,
    )
    return env
