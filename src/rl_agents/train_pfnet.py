#!/usr/bin/env python3

import argparse
import glob
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time
import wandb
from pathlib import Path
from pprint import pprint

from pfnetwork import pfnet
from environments.env_utils import datautils, render
from environments.envs.localize_env import LocalizeGibsonEnv
from pfnetwork.train import train_step, eval_step, prepare_data, WANDB_PROJECT, stack_loss_dicts, calc_metrics, vis_output, init_pfnet_model
from pfnetwork.arguments import parse_common_args, particle_std_to_covariance
from environments.suite_gibson import create_env

np.set_printoptions(suppress=True)



def pfnet_train(params):
    """
    A simple train for particle filter network

    :param params:
        parsed command-line arguments
    :return:
    """
    TRAIN_SUBDIR = "train_navagent"
    TEST_SUBDIR = "test_navagent"

    root_dir = os.path.expanduser(params.root_dir)
    train_dir = os.path.join(root_dir, TRAIN_SUBDIR)
    # eval_dir = os.path.join(root_dir, 'eval')

    # training data
    train_filenames = list(glob.glob(os.path.join(params.tfrecordpath, TRAIN_SUBDIR, '*.tfrecord')))
    train_ds = datautils.get_dataflow(train_filenames, params.batch_size, params.s_buffer_size, is_training=True, is_igibson=True)
    print(f'train data: {train_filenames}')

    # evaluation data
    eval_filenames = list(glob.glob(os.path.join(params.tfrecordpath, TEST_SUBDIR, '*.tfrecord')))
    eval_ds = datautils.get_dataflow(eval_filenames, params.batch_size, params.s_buffer_size, is_training=True, is_igibson=True)
    print(f'eval data: {eval_filenames}')

    pfnet_model = init_pfnet_model(params, is_igibson=True)

    # create igibson env which is used "only" to sample particles


    if params.epochs:
        env = create_env(params, pfnet_model=pfnet_model)
        optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate)
    else:
        env, optimizer = None, None

    # Define metrics
    train_loss = keras.metrics.Mean('train_loss', dtype=tf.float32)
    eval_loss = keras.metrics.Mean('eval_loss', dtype=tf.float32)

    for epoch in range(params.epochs):
        train_loss_dicts = []
        train_itr = train_ds.as_numpy_iterator()
        for _ in tqdm(range(params.num_train_batches), desc=f"Epoch {epoch}/{params.epochs}"):
            raw_train_record = next(train_itr)
            processed_data = datautils.transform_raw_record(env, raw_train_record, params)

            train_loss_dict, train_output, train_state = train_step(data=processed_data, model=pfnet_model, optimizer=optimizer, train_loss=train_loss, map_pixel_in_meters=params.map_pixel_in_meters)
            train_loss_dicts.append(train_loss_dict)

        if train_loss_dicts:
            train_loss_dicts = stack_loss_dicts(train_loss_dicts, 0, concat=True)
            train_metrics = calc_metrics(train_loss_dicts, prefix='train')
        else:
            train_metrics = {}


        eval_itr = eval_ds.as_numpy_iterator()
        eval_loss_dicts = []
        if len(eval_filenames):
            for _ in tqdm(range(params.num_eval_batches), desc=f"Epoch {epoch}/{params.epochs}"):
                raw_eval_record = next(eval_itr)
                processed_data = datautils.transform_raw_record(env, raw_eval_record, params)

                eval_loss_dict, eval_output, eval_state = eval_step(data=processed_data, model=pfnet_model, eval_loss=eval_loss, map_pixel_in_meters=params.map_pixel_in_meters)
                eval_loss_dicts.append(eval_loss_dict)

        if eval_loss_dicts:
            eval_loss_dicts = stack_loss_dicts(eval_loss_dicts, 0, concat=True)
            eval_metrics = calc_metrics(eval_loss_dicts, prefix='eval')
        else:
            eval_metrics = {}

        if epoch % 5 == 0:
            print("=====> saving trained model ")
            save_path = os.path.join(train_dir, f'chks/checkpoint_{epoch}_{eval_loss.result():03.3f}/pfnet_checkpoint')
            pfnet_model.save_weights(save_path)
            wandb.save(save_path)

        combined_metrics = dict(train_metrics, **eval_metrics)
        pprint(combined_metrics)
        wandb.log(combined_metrics, step=epoch)

        print(f'Epoch {epoch}, train loss: {train_loss.result():03.3f}, eval loss: {eval_loss.result():03.3f}')

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        eval_loss.reset_states()

    print('training finished')

    # test on a range of metrics
    # recreate dataset with batch size
    eval_ds = datautils.get_dataflow(eval_filenames, params.batch_size, params.s_buffer_size, is_training=True, is_igibson=True)
    params.batch_size = 1

    for distribution in ['uniform', 'gaussian']:
        for std_deviation in [0.15, 0.3, 0.5]:
            for num_particles in [30, 100, 300]:
                if env is not None:
                    env.close()
                    del env
                del pfnet_model

                if not params.pfnet_loadpath:
                    params.pfnet_loadpath = save_path

                params.init_particles_distr = distribution
                params.init_particles_std[0] = std_deviation
                params.init_particles_cov = particle_std_to_covariance(params.init_particles_std, map_pixel_in_meters=params.map_pixel_in_meters)
                params.num_particles = num_particles

                pfnet_model = init_pfnet_model(params, is_igibson=True)
                env = create_env(params, pfnet_model=pfnet_model)

                for agent in ['goalnav_agent', '', 'avoid_agent']:
                    params.agent = agent
                    test_name = f"test_{agent}_{distribution}_std{std_deviation}_p{num_particles}"
                    test_loss_dicts = []
                    if not agent:
                        itr = eval_ds.as_numpy_iterator() if len(eval_filenames) else train_ds.as_numpy_iterator()

                        for _ in tqdm(range(params.num_test_batches), desc=test_name):
                            raw_eval_record = next(itr)
                            processed_data = datautils.transform_raw_record(env, raw_eval_record, params)
                            test_loss_dict, eval_output, eval_state = eval_step(data=processed_data, model=pfnet_model,
                                                                                eval_loss=eval_loss,
                                                                                map_pixel_in_meters=params.map_pixel_in_meters)
                            test_loss_dicts.append(test_loss_dict)

                            if params.use_plot:
                                vis_output(env=env, output=eval_output, state=eval_state, data=processed_data)
                    else:
                        for _ in tqdm(range(min(params.num_test_batches, 100)), desc=test_name):
                            obs = env.reset()
                            for _ in range(params.trajlen - 1):
                                old_pose = env.get_robot_pose(env.robots[0].calc_state(), env.floor_map.shape)
                                action = datautils.select_action(agent=agent, params=params, obs=obs, env=env, old_pose=old_pose)
                                obs, reward, done, info = env.step(action)
                                test_loss_dicts.append(info)
                                if params.use_plot:
                                    env.render('human')

                    test_loss_dicts = stack_loss_dicts(test_loss_dicts, 0, concat=True)
                    test_metrics = calc_metrics(test_loss_dicts, prefix=test_name)
                    pprint(test_metrics)
                    wandb.log(test_metrics)

                    eval_loss.reset_states()

    env.close()


def check_scene_sizes():
    from PIL import Image
    import matplotlib.pyplot as plt
    # d = Path('/home/honerkam/repos/iGibson/igibson/data/g_dataset/')
    d = Path('/home/honerkam/repos/iGibson/igibson/data/data_100scenes/')
    scene_paths = d.iterdir()
    sizes = []
    for scene_path in scene_paths:
        if (not scene_path.is_dir()) or (scene_path.name in ['area1', 'Kinney', 'Gratz']):
            continue
        map_path = scene_path / 'floor_trav_0.png'
        if not map_path.exists():
            print([p.name for p in list(scene_path.iterdir())])
            assert False, map_path
        img = Image.open(map_path)
        bb = pfnet.PFCell.bounding_box(img)
        bb_size = (bb[1] - bb[0], bb[3] - bb[2])
        sizes.append(bb_size)
        print(scene_path.name, '\t', img.size, '\t', bb_size)
    sizes = np.stack(sizes)
    return sizes


if __name__ == '__main__':
    # parsed_params = parse_args()
    params = parse_common_args('igibson')
    params.use_tf_function = False

    run_name = Path(params.root_dir).name
    run = wandb.init(config=params, name=run_name, project=WANDB_PROJECT, sync_tensorboard=True, mode='disabled' if params.debug else 'online')

    pfnet_train(params)


# nohup python -u train_pfnet.py --root_dir=./run2 --tfrecordpath=/data/honerkam/pfnet_data/ --epochs=100 --obs_mode=rgb-depth --num_train_samples=4000 --num_eval_samples=500 --batch_size=12 --s_buffer_size=500 --pfnet_loadpath='' --learning_rate=5e-5 --init_particles_distr=gaussian --init_particles_std '0.15' '0.523599' --particles_range=100 --num_particles=30 --transition_std '0.' '0.' --resample=false --alpha_resample_ratio=0.5 --global_map_size 100 100 1 --window_scaler=1.0 --config_file=./configs/locobot_pfnet_nav.yaml --device_idx=2  --multiple_gpus=false --seed=42 > nohup.out &
# python -u train_pfnet.py --root_dir=./run_occupancy --tfrecordpath=/data2/honerkam/pfnet_data/ --epochs=100 --obs_mode=occupancy_grid --num_train_samples=4000 --num_eval_samples=500 --batch_size=12 --s_buffer_size=500 --pfnet_loadpath='' --learning_rate=5e-5 --init_particles_distr=gaussian --init_particles_std '0.15' '0.523599' --particles_range=100 --num_particles=30 --transition_std '0.' '0.' --resample=false --alpha_resample_ratio=0.5 --global_map_size 100 100 1 --window_scaler=1.0 --config_file=./configs/locobot_pfnet_nav.yaml --device_idx=7  --multiple_gpus=false --seed=42 
# TESTING COMMAND
# python -u train_pfnet.py --root_dir=./run2 --epochs=0 --obs_mode=rgb-depth --num_train_samples=4000 --num_eval_samples=500 --batch_size=1 --s_buffer_size=500 --learning_rate=5e-5 --init_particles_distr=uniform --init_particles_std '0.15' '0.523599' --particles_range=10 --num_particles=100 --transition_std 0 0 --resample=false --alpha_resample_ratio=0.5 --global_map_size 100 100 1 --window_scaler=1.0 --config_file=./configs/locobot_pfnet_nav.yaml --device_idx=2 --multiple_gpus=false --seed=42 --pfnet_loadpath=/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_95_0.075/pfnet_checkpoint

