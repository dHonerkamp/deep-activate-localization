#!/usr/bin/env python3


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.sharedctypes import Value

import os

from absl import flags
from absl import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

from pfnetwork.arguments import parse_common_args
from pfnetwork.train import WANDB_PROJECT, init_pfnet_model
from custom_agents.stable_baselines_utils import create_env, CustomCombinedExtractor3, MyWandbCallback, get_run_name, \
    get_logdir, MetricsCallback
from supervised_data import get_scene_ids


def make_sbl_env(rank, seed, params):
    def _init():
        pfnet_model = init_pfnet_model(params, is_igibson=True)
        env = create_env(params, pfnet_model=pfnet_model)

        env = Monitor(env)

        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init


def main(params, test_scenes=None):
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)

    if params.rl_architecture == 1:
        conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
        encoder_fc_layers = [512, 512]
        actor_fc_layers = [512, 512]
    elif params.rl_architecture == 2:
        conv_2d_layer_params = [(32, (3, 3), 2), (64, (3, 3), 2), (64, (3, 3), 2), (64, (3, 3), 2)]
        encoder_fc_layers = [512]
        actor_fc_layers = [512, 512]
    elif params.rl_architecture == 3:
        conv_2d_layer_params = [(32, (3, 3), 2), (64, (3, 3), 2), (64, (3, 3), 1), (64, (2, 2), 1)]
        encoder_fc_layers = [1024]
        actor_fc_layers = [512, 512]
    else:
        raise Value(params.rl_architecture)

    if params.num_parallel_environments > 1:
        env = SubprocVecEnv(
            [make_sbl_env(rank=i, seed=params.seed, params=params) for i in range(params.num_parallel_environments)])
    else:
        env = make_sbl_env(rank=0, seed=params.seed, params=params)()

    eval_env = None

    features_extractor_kwargs = dict(conv_2d_layer_params=conv_2d_layer_params,
                                     encoder_fc_layers=encoder_fc_layers)
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor3,
                         features_extractor_kwargs=features_extractor_kwargs,
                         net_arch=actor_fc_layers)

    model = SAC("MultiInputPolicy",
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                buffer_size=params.replay_buffer_capacity,
                gamma=params.gamma,
                learning_rate=params.actor_learning_rate,
                batch_size=params.rl_batch_size,
                seed=params.seed,
                learning_starts=params.initial_collect_steps,
                train_freq=params.collect_steps_per_iteration,
                ent_coef=params.ent_coef,
                tensorboard_log=os.path.join(params.root_dir, 'train'))
    if params.num_iterations:
        cb = CallbackList([MetricsCallback(),
                           MyWandbCallback(model_save_path=Path(params.root_dir) / 'train' / 'ckpts',
                                           model_save_freq=params.eval_interval)])
        model.learn(total_timesteps=params.num_iterations,
                    log_interval=4,
                    eval_freq=params.eval_interval,
                    n_eval_episodes=params.num_eval_episodes,
                    callback=cb,
                    eval_env=eval_env)
        model.save("sac_rl_agent")


if __name__ == '__main__':
    params = parse_common_args('igibson', add_rl_args=True)
    params.agent = 'rl'
    # run_name = Path(params.root_dir).name

    if params.scene_id == "all":
        train_scenes, test_scenes = get_scene_ids(params.global_map_size)
        params.scene_id = train_scenes
    else:
        assert False, "Sure you want to train on a single scene?"

    run_name = get_run_name(params)
    params.root_dir = str(get_logdir(run_name))

    run = wandb.init(config=params, name=run_name, project=WANDB_PROJECT, sync_tensorboard=True,
                     mode='disabled' if params.debug else 'online')

    main(params, test_scenes=test_scenes)

# python -u train_eval.py --root_dir 'train_output' --eval_only=False --num_iterations 3000 --initial_collect_steps 500 --use_parallel_envs=True --collect_steps_per_iteration 1 --num_parallel_environments 1 --num_parallel_environments_eval 1 --replay_buffer_capacity 1000 --train_steps_per_iteration 1 --batch_size 16 --num_eval_episodes 10 --eval_interval 500 --device_idx=0 --seed 100 --pfnet_loadpath=/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint
# nohup python -u train_eval.py --root_dir=train_output --eval_only=False --num_iterations=3000 --initial_collect_steps=500 --use_parallel_envs=no --collect_steps_per_iteration=1 --replay_buffer_capacity=1000 --rl_batch_size=16 --num_eval_episodes=10 --eval_interval=500 --device_idx=2 --seed=100 --custom_output rgb_obs depth_obs likelihood_map --pfnet_loadpath=/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint > nohup_rl.out &
# nohup python -u sbl_train_eval.py --device_idx "2" --num_parallel_environments 8 --custom_output "rgb_obs" "depth_obs" "likelihood_map" "task_obs" --replay_buffer_capacity "50000" --initial_collect_steps "0" --resample yes --alpha_resample_ratio 0.5 --num_particles 500 --eval_interval "500" --pfnet_loadpath "/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint" > nohup_sbl.out &
# nohup python -u sbl_train_eval.py --device_idx "1" --scene_id all --num_parallel_environments 7 --custom_output "rgb_obs" "depth_obs" "task_obs" "likelihood_map" --global_map_size 1000 1000 1 --replay_buffer_capacity "50000" --initial_collect_steps "0" --resample yes --alpha_resample_ratio 0.5 --num_particles 500 --eval_interval "500" --pfnet_loadpath "/home/honerkam/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000/train_navagent_below1000/chks/checkpoint_65_0.475/pfnet_checkpoint" --init_particles_distr uniform --particles_range 10 --trajlen 50 > nohup_sbl4.out &
# nohup python -u sbl_train_eval.py --device_idx "1" --actor_learning_rate 0.0001 --scene_id all --num_parallel_environments 7 --custom_output "rgb_obs" "depth_obs" "task_obs" "likelihood_map" --global_map_size 1000 1000 1 --replay_buffer_capacity 100000 --initial_collect_steps "0" --resample yes --alpha_resample_ratio 0.5 --num_particles 500 --eval_interval "500" --pfnet_loadpath "/home/honerkam/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000/train_navagent_below1000/chks/checkpoint_65_0.475/pfnet_checkpoint" --init_particles_distr uniform --particles_range 10 --trajlen 50 > nohup_sbl7.out &
# nohup python -u sbl_train_eval.py --device_idx "1" --scene_id all --num_parallel_environments 7 --custom_output "rgb_obs" "depth_obs" "task_obs" "likelihood_map" --global_map_size 1000 1000 1 --replay_buffer_capacity "50000" --initial_collect_steps "0" --resample yes --alpha_resample_ratio 0.5 --num_particles 500 --eval_interval "500" --pfnet_loadpath "/home/honerkam/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000_gaussian/train_navagent_below1000/chks/checkpoint_95_0.755/pfnet_checkpoint" --init_particles_distr uniform --particles_range 10 --trajlen 50 > nohup_sbl4.out &
# nohup python -u sbl_train_eval.py --device_idx "1" --scene_id all --num_parallel_environments 7 --custom_output "task_obs" "likelihood_map" "occupancy_grid" "depth_obs" "rgb_obs" --obs_mode "occupancy_grid" --global_map_size 1000 1000 1 --replay_buffer_capacity "50000" --initial_collect_steps "0" --resample yes --alpha_resample_ratio 0.5 --num_particles 250 --eval_interval "500" --pfnet_loadpath "/home/honerkam/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000_lidar030/train_navagent_below1000/chks/checkpoint_95_0.157/pfnet_checkpoint" --init_particles_distr uniform --particles_range 10 --trajlen 50 --rl_architecture 2 > nohup_sbl.out &
