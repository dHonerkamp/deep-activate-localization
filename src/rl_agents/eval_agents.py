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
from stable_baselines3.common.vec_env import SubprocVecEnv
from pprint import pprint
import copy
import sys
from tqdm import tqdm
from argparse import Namespace

from pfnetwork.arguments import parse_common_args, particle_std_to_covariance
from pfnetwork.train import WANDB_PROJECT, stack_loss_dicts, calc_metrics
from sbl_train_eval import make_sbl_env
from custom_agents.stable_baselines_utils import DotDict
from environments.env_utils import datautils
from custom_agents.stable_baselines_utils import get_run_name, get_logdir
from supervised_data import get_scene_ids
from environments import suite_gibson
from render_paper import Trajectory
from pfnetwork import pfnet



def evaluate(params, distribution, std_deviation, num_particles, particles_range, resample, alpha_resample_ratio, unseen: bool):
    params.init_particles_distr = distribution
    params.init_particles_std[0] = std_deviation
    params.init_particles_cov = particle_std_to_covariance(params.init_particles_std, map_pixel_in_meters=params.map_pixel_in_meters)
    params.num_particles = num_particles
    params.particles_range = particles_range
    params.resample = resample
    params.alpha_resample_ratio = alpha_resample_ratio
    
    if (not params.resume_id) and ("obstacle_obs" not in params.custom_output):
        params.custom_output.append("obstacle_obs")
        
    # if params.scene_id == 'all':
    # TODO: evaluate on test scenes?
    train_scenes, test_scenes = get_scene_ids(params.global_map_size)
    assert len(train_scenes), (params.global_map_size, train_scenes)
    assert len(test_scenes), (params.global_map_size, test_scenes)
    if unseen:
        params.scene_id = test_scenes
    else:
        params.scene_id = train_scenes
        
    env = make_sbl_env(rank=0, seed=params.seed, params=params)()
    
    
    
    # if params.num_parallel_environments > 1:
    #     env = SubprocVecEnv([make_sbl_env(rank=i, seed=params.seed, params=Namespace(**params)) for i in range(params.num_parallel_environments)])
    # else:
    #     env = make_sbl_env(rank=0, seed=params.seed, params=params)()
    #     env = DummyVecEnv(env)
    
    if params.agent == 'rl':
        model_file = wandb.restore(params.resume_model_name if params.resume_model_name else "model.zip")
        agent = SAC.load(model_file.name, env=env)
    else:
        agent = params.agent

    test_name = f"test_p{num_particles}{distribution}std{std_deviation}rng{particles_range}_r{resample}a{alpha_resample_ratio}{'_unseen' if unseen else ''}"
    test_loss_dicts = []
    videos = []
    trajectories = []
    for i in tqdm(range(min(params.num_eval_episodes, 100))):
        obs = env.reset()
        traj = Trajectory(scene_id=env.config["scene_id"])
        traj.add(observation=obs, 
                 gt_pose=env.curr_gt_pose, 
                 est_pose=env.curr_est_pose,
                 robot_position=env.robots[0].robot_body.get_position(),
                 robot_orientation=env.robots[0].robot_body.get_orientation(),
                 likelihood_map=pfnet.PFCell.get_likelihood_map(particles=env.curr_pfnet_state[0],
                                                                particle_weights=env.curr_pfnet_state[1],
                                                                floor_map=env.floor_map))
        
        if env.last_video_path:
            videos.append(wandb.Video(env.last_video_path))
        
        done = False
        
        while not done:
            if isinstance(agent, str):
                old_pose = env.get_robot_pose(env.robots[0].calc_state())
                action = datautils.select_action(agent=agent, params=params, obs=obs, env=env, old_pose=old_pose)
            else:
                action, _ = agent.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            test_loss_dicts.append(info)
            if (i < 15) and (params.use_plot or params.store_plot):
                # env.render('human', info=info, observation=obs)
                
                # TODO: also needs to have set/loaded the same scene
                # hr_env.robots[0].set_position_orientation(env.robots[0].robot_body.get_position(), env.robots[0].robot_body.get_orientation())
                # hr_state = hr_env.get_state()
                # hr_obs = hr_env.process_state(hr_state)
                # env.render('human', info=info, observation=obs)
                
                # from matplotlib import pyplot as plt
                # f, ax = plt.subplots(1, 1)
                # # ax.imshow(self.eps_obs['depth'][-1])
                # ax.imshow(hr_state['depth'])
                # f.savefig('test2.png')
                
                traj.add(observation=obs, 
                         gt_pose=env.curr_gt_pose, 
                         est_pose=env.curr_est_pose, 
                         robot_position=env.robots[0].robot_body.get_position(),
                         robot_orientation=env.robots[0].robot_body.get_orientation(),
                         likelihood_map=pfnet.PFCell.get_likelihood_map(particles=env.curr_pfnet_state[0],
                                                                        particle_weights=env.curr_pfnet_state[1],
                                                                        floor_map=env.floor_map))
                
                if done:
                    trajectories.append(traj)
                    # figures = traj.store_video(out_folder=env.out_folder, episode_number=i)
    
    test_loss_dicts = stack_loss_dicts(test_loss_dicts, 0, concat=True)
    for k in test_loss_dicts.keys():
        test_loss_dicts[k] = tf.reshape(test_loss_dicts[k], [min(params.num_eval_episodes, 100), -1])
    test_metrics = calc_metrics(test_loss_dicts, prefix=test_name)
    pprint(test_metrics)
    for i, video in enumerate(videos[:5]):
        test_metrics[f'{test_name}/video{i}'] = video
    wandb.log(test_metrics, commit=True)
    print('done')
    
    env.close()
    # NOTE: just to have high-res images for the paper
    if (params.use_plot or params.store_plot):
        if isinstance(params, Namespace):
            params = vars(params)
        hr_params = DotDict(copy.deepcopy(dict(params)))
        hr_params.custom_output.remove('likelihood_map')
        hr_params['high_res'] = True
        # hr_env = make_sbl_env(rank=0, seed=hr_params.seed, params=hr_params)()
        hr_env = suite_gibson.create_env(hr_params, pfnet_model=None, do_wrap_env=False)
        hr_env.scene_ids = []

        hr_env.reset()
        
        for n, traj in enumerate(trajectories):
            hr_env.config["scene_id"] = traj.scene_id
            hr_env.reload_model(traj.scene_id)
            for i in range(len(traj.observation)):
                hr_env.robots[0].set_position_orientation(traj.robot_position[i], traj.robot_orientation[i])
                hr_state = hr_env.get_state()
                # hr_obs = hr_env.process_state(hr_state)
                
                traj.observation[i]['rgb_obs'] = hr_state['rgb']        
                traj.observation[i]['depth_obs'] = hr_state['depth']   
                traj.observation[i]['occupancy_grid'] = hr_state['occupancy_grid']
                
                # from matplotlib import pyplot as plt
                # f, ax = plt.subplots(1, 1)
                # # ax.imshow(self.eps_obs['depth'][-1])
                # ax.imshow(hr_state['rgb'])
                # f.savefig('test2.png')
                     
            figures = traj.store_video(out_folder=env.out_folder, episode_number=n)
    
    
    
    print("donesies 1")
    
def main(params):
    alpha_resample = 0.5
    i = 0
    for unseen in [True, False]:
        for (dist, std, num, rng, resample, alpha) in [("gaussian", 0.3, 300, 100, True, alpha_resample),
                                                       ("uniform", 0.3, 500, 10, True, alpha_resample),
                                                       ("uniform", 0.3, 3000, 1000, True, alpha_resample),
                                                       # ("uniform", 0.15, 500, 10, True, alpha_resample), 
                                                       # ("uniform", 0.15, 500, 10, False, alpha_resample),
                                                       ]:
            evaluate(params, distribution=dist, std_deviation=std, num_particles=num, particles_range=rng, resample=resample, alpha_resample_ratio=alpha, unseen=unseen)
            i += 1
    

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
        for p in ['num_particles', 'transition_std', 'resample', 'alpha_resample_ratio', 'init_particles_distr', 'init_particles_std', 
                  'use_plot', 'store_plot', "scene_id", "trajlen"]:
            if p in cl_args:
                wandb_params[p] = params.__getattribute__(p)
                
        # always override certain values
        for p in ['num_parallel_environments', 'num_eval_episodes', "resume_model_name", 'resume_id', 'action_timestep', 'loop']:
            wandb_params[p] = params.__getattribute__(p)
        
        # backwards compatibility for new keys
        for p in ['reward', 'rl_architecture', 'collision_reward_weight']:
            if not wandb_params.get(p, None):
                wandb_params[p] = params.__getattribute__(p)
        if not wandb_params.get('observe_steps', None):
            wandb_params['observe_steps'] = False
            
        params = wandb_params
        # backwards compatibility 
        params.agent = 'rl'
        # run_name = get_run_name(params)
        params.root_dir = str(get_logdir(params.resume_id))
    else:
        # run_name = Path(params.root_dir).name
        run_name = get_run_name(params)
        params.root_dir = str(get_logdir(run_name))
        run = wandb.init(**common_args, config=params, name=run_name, mode='disabled' if params.debug else 'online')
    
    main(params)
    
    # wandb.gym.monitor()

# nohup python -u eval_agents.py --device_idx 3 --agent avoid_agent --trajlen 50 --custom_output "rgb_obs" "depth_obs" "likelihood_map" "obstacle_obs" "task_obs" --eval_only --use_plot --store_plot --num_eval_episodes 50 --scene_id all --global_map_size 1000 1000 1 --pfnet_loadpath "/home/honerkam/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000_lidar030/train_navagent_below1000/chks/checkpoint_95_0.157/pfnet_checkpoint" --obs_mode occupancy_grid > nohup_eval.out &
# ON MY DESKTOP
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python -u eval_agents.py --device_idx 0 --agent manual_agent --custom_output "rgb_obs" "depth_obs" "likelihood_map" "obstacle_obs" "task_obs" occupancy_grid --eval_only --use_plot --store_plot=no --num_eval_episodes 50 --pfnet_loadpath "/home/daniel/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000_lidar030/train_navagent_below1000/chks/checkpoint_95_0.157/pfnet_checkpoint" --env_mode gui --obs_mode "occupancy_grid"



# python -u  eval_agents.py --device_idx 2 --custom_output rgb_obs depth_obs likelihood_map obstacle_obs occupancy_grid --obs_mode occupancy_grid --pfnet_loadpath /home/honerkam/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000_lidar030/train_navagent_below1000/chks/checkpoint_95_0.157/pfnet_checkpoint --agent goalnav_agent --resume_id zev67g1g --eval_only --use_plot --store_plot --num_eval_episodes 15 --scene_id all --global_map_size 1000 1000 1