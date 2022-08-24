#!/usr/bin/env python3


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from distutils.log import log

import os
import time

# from absl import app
from absl import flags
from absl import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor

from pfnetwork.arguments import parse_common_args
from pfnetwork.train import WANDB_PROJECT, init_pfnet_model
from environments import suite_gibson
from custom_agents.stable_baselines_utils import CustomCombinedExtractor, MyWandbCallback, get_run_name, get_logdir

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_particles', 500, 'Number of particles in Particle Filter.')
flags.DEFINE_boolean('resample', True, 'Resample particles in Particle Filter. Possible values: true / false.')
flags.DEFINE_float('alpha_resample_ratio', 0.5, 'Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true. Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling.')
flags.DEFINE_list('transition_std', [0.02, 0.0872665], 'Standard deviations for transition model. Values: translation std (meters), rotation std (radians)')
FLAGS = flags.FLAGS



def make_sbl_env(rank, seed, params):    
    def _init():
        pfnet_model = init_pfnet_model(params, is_igibson=True)
        env = suite_gibson.create_env(params, pfnet_model=pfnet_model, do_wrap_env=False)
        
        env = Monitor(env)
        
        env.seed(seed + rank)
        
        return env
    set_random_seed(seed)
    return _init
    

def main(params):
    tf.compat.v1.enable_v2_behavior()
    # tf.debugging.enable_check_numerics()    # error out inf or NaN
    logging.set_verbosity(logging.INFO)

    conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
    encoder_fc_layers = [512, 512]
    actor_fc_layers = [512, 512]
    # critic_obs_fc_layers = [512, 512]
    # critic_action_fc_layers = [512, 512]
    # critic_joint_fc_layers = [512, 512]

    # print('==================================================')
    # for k, v in FLAGS.flag_values_dict().items():
    #     print(k, v)
    # print('conv_1d_layer_params', conv_1d_layer_params)
    # print('conv_2d_layer_params', conv_2d_layer_params)
    # print('encoder_fc_layers', encoder_fc_layers)
    # print('actor_fc_layers', actor_fc_layers)
    # print('critic_obs_fc_layers', critic_obs_fc_layers)
    # print('critic_action_fc_layers', critic_action_fc_layers)
    # print('critic_joint_fc_layers', critic_joint_fc_layers)
    # print('==================================================')

    # pfnet_model = init_pfnet_model(params, is_igibson=True)
    # env = suite_gibson.create_env(params, pfnet_model=pfnet_model, do_wrap_env=False)
    
    if params.num_parallel_environments > 1:
        env = SubprocVecEnv([make_sbl_env(rank=i, seed=params.seed, params=params) for i in range(params.num_parallel_environments)])
        eval_env = None
    else:
        env = make_sbl_env(rank=0, seed=params.seed, params=params)()
        eval_env = env

    features_extractor_kwargs = dict(conv_2d_layer_params=conv_2d_layer_params,
                                                        encoder_fc_layers=encoder_fc_layers)
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
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
                tensorboard_log=os.path.join(params.root_dir, 'train'))
    if params.num_iterations:
        model.learn(total_timesteps=params.num_iterations, 
                    log_interval=4,
                    eval_freq=params.eval_interval,
                    n_eval_episodes=params.num_eval_episodes,
                    callback=MyWandbCallback(model_save_path=Path(params.root_dir) / 'train' / 'ckpts', 
                                             model_save_freq=params.eval_interval),
                    eval_env=eval_env)
        model.save("sac_rl_agent")
        
        
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)


if __name__ == '__main__':
    params = parse_common_args('igibson', add_rl_args=True)
    params.agent = 'rl'
    # run_name = Path(params.root_dir).name
    
    run_name = get_run_name(params)
    params.root_dir = str(get_logdir(run_name))

    run = wandb.init(config=params, name=run_name, project=WANDB_PROJECT, sync_tensorboard=True, mode='disabled' if params.debug else 'online')

    main(params)



# python -u train_eval.py --root_dir 'train_output' --eval_only=False --num_iterations 3000 --initial_collect_steps 500 --use_parallel_envs=True --collect_steps_per_iteration 1 --num_parallel_environments 1 --num_parallel_environments_eval 1 --replay_buffer_capacity 1000 --train_steps_per_iteration 1 --batch_size 16 --num_eval_episodes 10 --eval_interval 500 --device_idx=0 --seed 100 --pfnet_loadpath=/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint
# nohup python -u train_eval.py --root_dir=train_output --eval_only=False --num_iterations=3000 --initial_collect_steps=500 --use_parallel_envs=no --collect_steps_per_iteration=1 --replay_buffer_capacity=1000 --rl_batch_size=16 --num_eval_episodes=10 --eval_interval=500 --device_idx=2 --seed=100 --custom_output rgb_obs depth_obs likelihood_map --pfnet_loadpath=/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint > nohup_rl.out &
# nohup python -u sbl_train_eval.py --device_idx "2" --num_parallel_environments 8 --custom_output "rgb_obs", "depth_obs", "likelihood_map", "task_obs" --replay_buffer_capacity "50000" --initial_collect_steps "0" --resample yes --alpha_resample_ratio 0.5 --num_particles 500 --eval_interval "500" --pfnet_loadpath "/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint" > nohup_sbl.out &
# nohup python -u sbl_train_eval.py --device_idx "2" --num_parallel_environments 8 --custom_output "rgb_obs", "depth_obs", "likelihood_map", "task_obs" --replay_buffer_capacity "50000" --initial_collect_steps "0" --resample yes --alpha_resample_ratio 0.5 --num_particles 500 --eval_interval "500" --pfnet_loadpath "/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint" --init_particles_distr uniform --particles_range 10 --trajlen 50 > nohup_sbl4.out &
