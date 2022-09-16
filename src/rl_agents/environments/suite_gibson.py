#!/usr/bin/env python3

from .envs.navigate_env import NavigateGibsonEnv
from .envs.localize_env import LocalizeGibsonEnv
# import gin
# custom tf_agents
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers


def create_env(params, pfnet_model, do_wrap_env: bool = False):
    env = LocalizeGibsonEnv(
        config_file=params.config_file,
        scene_id=params.scene_id,
        mode=params.env_mode,
        # use_tf_function=params.use_tf_function,
        # init_pfnet=params.init_env_pfnet,
        action_timestep=params.action_timestep,
        physics_timestep=params.physics_timestep,
        device_idx=params.device_idx,
        pfnet_model=pfnet_model,
        pf_params=params,
    )
    # env.reset()

    if do_wrap_env:
        raise NotImplementedError()
        # discount = env.config['discount_factor']
        # assert params.gamma == env.config['discount_factor']
        # max_episode_steps = env.config['max_step']

        # env = wrap_env(env,
        #                discount=params.gamma,
        #                max_episode_steps=max_episode_steps,
        #                gym_env_wrappers=(),
        #                time_limit_wrapper=wrappers.TimeLimit,
        #                env_wrappers=(),
        #                spec_dtype_map=None,
        #                auto_reset=True)
        
    return env


# @gin.configurable
# def load(config_file,
#          model_id=None,
#          env_mode='headless',
#          use_tf_function=True,
#          init_pfnet=False,
#          is_localize_env=True,
#          action_timestep=1.0 / 10.0,
#          physics_timestep=1.0 / 40.0,
#          device_idx=0,
#          gym_env_wrappers=(),
#          env_wrappers=(),
#          spec_dtype_map=None):
#     if is_localize_env:
#         env = LocalizeGibsonEnv(config_file=config_file,
#                                 scene_id=model_id,
#                                 mode=env_mode,
#                                 use_tf_function=use_tf_function,
#                                 init_pfnet=init_pfnet,
#                                 action_timestep=action_timestep,
#                                 physics_timestep=physics_timestep,
#                                 device_idx=device_idx)
#     else:
#         env = NavigateGibsonEnv(config_file=config_file,
#                                 scene_id=model_id,
#                                 mode=env_mode,
#                                 action_timestep=action_timestep,
#                                 physics_timestep=physics_timestep,
#                                 device_idx=device_idx)

#     discount = env.config.get('discount_factor', 0.99)
#     max_episode_steps = env.config.get('max_step', 500)

#     return wrap_env(
#         env,
#         discount=discount,
#         max_episode_steps=max_episode_steps,
#         gym_env_wrappers=gym_env_wrappers,
#         time_limit_wrapper=wrappers.TimeLimit,
#         env_wrappers=env_wrappers,
#         spec_dtype_map=spec_dtype_map,
#         auto_reset=True
#     )


# @gin.configurable
# def wrap_env(env,
#              discount=1.0,
#              max_episode_steps=0,
#              gym_env_wrappers=(),
#              time_limit_wrapper=wrappers.TimeLimit,
#              env_wrappers=(),
#              spec_dtype_map=None,
#              auto_reset=True):
#     for wrapper in gym_env_wrappers:
#         env = wrapper(env)
#     env = gym_wrapper.GymWrapper(
#         env,
#         discount=discount,
#         spec_dtype_map=spec_dtype_map,
#         match_obs_space_dtype=True,
#         auto_reset=auto_reset,
#         simplify_box_bounds=True
#     )

#     if max_episode_steps > 0:
#         env = time_limit_wrapper(env, max_episode_steps)

#     for wrapper in env_wrappers:
#         env = wrapper(env)

#     return env
