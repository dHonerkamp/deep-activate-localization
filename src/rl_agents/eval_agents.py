#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import time

# from absl import flags
# from absl import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
import wandb
from stable_baselines3 import SAC
from pprint import pprint
import copy
import sys
from tqdm import tqdm

from pfnetwork.arguments import parse_common_args, particle_std_to_covariance
from pfnetwork.train import WANDB_PROJECT, stack_loss_dicts, calc_metrics
from sbl_train_eval import make_sbl_env
from custom_agents.stable_baselines_utils import DotDict
from environments.env_utils import datautils
from custom_agents.stable_baselines_utils import get_run_name, get_logdir


def eval(params, distribution, std_deviation, num_particles, particles_range, resample, alpha_resample_ratio):
    params.init_particles_distr = distribution
    params.init_particles_std[0] = std_deviation
    params.init_particles_cov = particle_std_to_covariance(params.init_particles_std, map_pixel_in_meters=params.map_pixel_in_meters)
    params.num_particles = num_particles
    params.particles_range = particles_range
    params.resample = resample
    params.alpha_resample_ratio = alpha_resample_ratio
    
    if (not params.resume_id) and ("obstacle_obs" not in params.custom_output):
        params.custom_output.append("obstacle_obs")
    env = make_sbl_env(rank=0, seed=params.seed, params=params)()
    
    if params.agent == 'rl':
        model_file = wandb.restore('model.zip')
        agent = SAC.load(model_file.name, env=env)
    else:
        agent = params.agent

    test_name = f"test_p{num_particles}{distribution}std{std_deviation}rng{particles_range}_r{resample}a{alpha_resample_ratio}"
    test_loss_dicts = []
    videos = []
    for _ in tqdm(range(min(params.num_eval_episodes, 100))):
        obs = env.reset()
        
        if env.last_video_path:
            videos.append(wandb.Video(env.last_video_path))
        
        done = False
        
        while not done:
            if isinstance(agent, str):
                old_pose = env.get_robot_pose(env.robots[0].calc_state(), env.floor_map.shape)
                action = datautils.select_action(agent=agent, params=params, obs=obs, env=env, old_pose=old_pose)
            else:
                action, _ = agent.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            test_loss_dicts.append(info)
            if params.use_plot or params.store_plot:
                env.render('human')
    
    test_loss_dicts = stack_loss_dicts(test_loss_dicts, 0, concat=True)
    test_metrics = calc_metrics(test_loss_dicts, prefix=test_name)
    pprint(test_metrics)
    for i, video in enumerate(videos[:5]):
        test_metrics[f'{test_name}/video{i}'] = video
    wandb.log(test_metrics, commit=True)
    print('done')
    
    
def main(params):
    alpha_resample = 0.5
    for (dist, std, num, rng, resample, alpha) in [("uniform", 0.15, 500, 10, True, alpha_resample), 
                                                   ("uniform", 0.15, 500, 10, False, alpha_resample),
                                                   ("uniform", 0.15, 500, 100, True, alpha_resample)]:
        eval(params, distribution=dist, std_deviation=std, num_particles=num, particles_range=rng, resample=resample, alpha_resample_ratio=alpha)
    
    
if __name__ == '__main__':
    # tf.compat.v1.enable_v2_behavior()

    params = parse_common_args('igibson', add_rl_args=True)

    common_args = dict(project=WANDB_PROJECT, 
                       sync_tensorboard=True)
    
    if params.resume_id:
        run = wandb.init(**common_args, id=params.resume_id, resume='must')
        
        # allow to override certain args with command line arguments
        wandb_params = DotDict(copy.deepcopy(dict(wandb.config)))
        raw_args = sys.argv
        cl_args = [k.replace('-', '').replace(" ", "=").split('=')[0] for k in raw_args]
        for p in ['resume_id', 'num_particles', 'transition_std', 'resample', 'alpha_resample_ratio', 'init_particles_distr', 'init_particles_std', 
                  'use_plot', 'store_plot']:
            if p in cl_args:
                wandb_params[p] = params.__getattribute__(p)
            
        params = wandb_params
        # backwards compatibility 
        params.agent = 'rl'
        run_name = get_run_name(params)
        params.root_dir = str(get_logdir(run_name))
    else:
        # run_name = Path(params.root_dir).name
        run_name = get_run_name(params)
        params.root_dir = str(get_logdir(run_name))
        run = wandb.init(**common_args, config=params, name=run_name, mode='disabled' if params.debug else 'online')
    
    main(params)
    
    # wandb.gym.monitor()

# nohup python -u eval_agents.py --device_idx 1 --agent avoid_agent --resume_id 3p16emk2 --custom_output "rgb_obs" "depth_obs", "likelihood_map", "obstacle_obs", "task_obs" --eval_only --use_plot --store_plot --num_eval_episodes 50 --pfnet_loadpath /home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint > nohup_eval.out &
