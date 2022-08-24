#!/usr/bin/env python3

# from absl import app
# from absl import flags
from absl import logging
import argparse
import numpy as np
import os
import random
import tensorflow as tf
from tqdm import tqdm

# import custom tf_agents
from environments.env_utils import datautils
from environments.envs.localize_env import LocalizeGibsonEnv
from pfnetwork.arguments import parse_common_args

# define testing parameters
# flags.DEFINE_string(
#     name='filename',
#     default='./test.tfrecord',
#     help='The tf record.'
# )
# flags.DEFINE_integer(
#     name='num_records',
#     default=10,
#     help='The number of episode data.'
# )
# flags.DEFINE_integer(
#     name='seed',
#     default=100,
#     help='Fix the random seed'
# )
# flags.DEFINE_string(
#     name='agent',
#     default='avoid_agent',
#     help='Agent Behavior'
# )

# define igibson env parameters
# flags.DEFINE_string(
#     name='config_file',
#     default=os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         'configs',
#         'locobot_pfnet_nav.yaml'
#     ),
#     help='Config file for the experiment'
# )
# flags.DEFINE_string(
#     name='scene_id',
#     default=None,
#     help='Environment scene'
# )
# flags.DEFINE_string(
#     name='env_mode',
#     default='headless',
#     help='Environment render mode'
# )
# flags.DEFINE_string(
#     name='obs_mode',
#     default='rgb-depth',
#     help='Observation input type. Possible values: rgb / depth / rgb-depth / occupancy_grid.'
# )
# flags.DEFINE_list(
#     name='custom_output',
#     default=['rgb_obs', 'depth_obs', 'occupancy_grid', 'floor_map', 'kmeans_cluster', 'likelihood_map'],
#     help='A comma-separated list of env observation types.'
# )
# flags.DEFINE_float(
#     name='action_timestep',
#     default=1.0 / 10.0,
#     help='Action time step for the simulator'
# )
# flags.DEFINE_float(
#     name='physics_timestep',
#     default=1.0 / 40.0,
#     help='Physics time step for the simulator'
# )
# flags.DEFINE_integer(
#     name='gpu_num',
#     default=0,
#     help='GPU id for graphics/computation'
# )
# flags.DEFINE_boolean(
#     name='is_discrete',
#     default=False,
#     help='Whether to use discrete/continuous actions'
# )
# flags.DEFINE_float(
#     name='velocity',
#     default=1.0,
#     help='Velocity of Robot'
# )
# flags.DEFINE_integer(
#     name='max_step',
#     default=150,
#     help='The maimum number of episode steps.'
# )

# define pfNet env parameters
# flags.DEFINE_boolean(
#     name='init_env_pfnet',
#     default=False,
#     help='Whether to initialize particle filter net'
# )

# FLAGS = flags.FLAGS


# def get_shortest_path(env, start, goal):
#     path, dist = env.scene.get_shortest_path(env.task.floor_num, start[:2], goal[:2], entire_path=True)
#     return path


def collect_data(env, params, filename='./test.tfrecord', num_records=10):
    """
    Run the gym environment and collect the required stats
    :param env: igibson env instance
    :param params: parsed parameters
    :param filename: tf record file name
    :param num_records: number of records(episodes) to collect
    :return dict: episode stats data containing:
        odometry, true poses, observation, particles, particles weights, floor map
    """

    with tf.io.TFRecordWriter(filename) as writer:
        for i in tqdm(range(num_records)):
            # print(f'episode: {i}')
            episode_data = datautils.gather_episode_stats(env, params, sample_particles=False)
            record = datautils.serialize_tf_record(episode_data)
            writer.write(record)

    print(f'Collected successfully in {filename}')

    # sanity check
    ds = datautils.get_dataflow([filename], batch_size=1, s_buffer_size=100, is_training=False, is_igibson=True)
    data_itr = ds.as_numpy_iterator()
    for idx in range(num_records):
        parsed_record = next(data_itr)
        batch_sample = datautils.transform_raw_record(env, parsed_record, params)


def main():
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    # tf.debugging.enable_check_numerics()  # error out inf or NaN

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device_idx)
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # set random seeds
    # random.seed(FLAGS.seed)
    # np.random.seed(FLAGS.seed)
    # tf.random.set_seed(FLAGS.seed)

    params = parse_common_args(env_name='igibson', collect_data=True)


    if os.path.exists(params.filename):
        print(f'File {os.path.abspath(params.filename)} already exists !!!')
        return

    env = LocalizeGibsonEnv(
        config_file=params.config_file,
        scene_id=params.scene_id,
        mode=params.env_mode,
        # use_tf_function=False,
        pfnet_model=None,
        pf_params=params,
        action_timestep=params.action_timestep,
        physics_timestep=params.physics_timestep,
        device_idx=params.device_idx
    )
    # HACK: override value from config file
    # FLAGS.max_step = env.config.get('max_step', 500)
    # FLAGS.is_discrete = env.config.get("is_discrete", False)
    # FLAGS.velocity = env.config.get("velocity", 1.0)

    # print('==================================================')
    # for k, v in params.items():
    #     print(k, v)
    # print('==================================================')

    # argparser = argparse.ArgumentParser()
    # params = argparser.parse_args([])

    # For the igibson maps, originally each pixel represents 0.01m, and the center of the image correspond to (0,0)
    # params.map_pixel_in_meters = 0.01
    # in igibson we work with rescaled 0.01m to 0.1m maps to sample robot poses
    # params.trav_map_resolution = 0.1
    # params.loop = 6
    # params.agent = FLAGS.agent
    # params.trajlen = FLAGS.max_step // params.loop
    # params.max_lin_vel = env.config.get("linear_velocity")
    # params.max_ang_vel = env.config.get("angular_velocity")
    # params.global_map_size = np.array([400, 400, 1])
    # # params.obs_mode = FLAGS.obs_mode
    # params.batch_size = 1
    # params.num_particles = 10
    # # params.init_particles_distr = 'gaussian'
    # # particle_std = np.array([0.15, 0.523599])
    # transition_std = np.array([0.02, 0.0872665])
    # transition_std[0] = (transition_std[0] / params.map_pixel_in_meters) * params.trav_map_resolution  # convert meters to pixels and rescale to trav map resolution
    # params.particle_std[0] = (params.particle_std[0] / params.map_pixel_in_meters) * params.trav_map_resolution  # convert meters to pixels and rescale to trav map resolution
    # particle_std2 = np.square(params.particle_std)  # variance
    # params.init_particles_cov = np.diag(particle_std2[(0, 0, 1),])
    # params.transition_std = transition_std
    # # params.particles_range = 100

    # # compute observation channel dim
    # if params.obs_mode == 'rgb-depth':
    #     params.obs_ch = 4
    # elif params.obs_mode == 'rgb':
    #     params.obs_ch = 3
    # elif params.obs_mode == 'depth' or params.obs_mode == 'occupancy_grid':
    #     params.obs_ch = 1
    # else:
    #     raise ValueError

    print(params)
    collect_data(env, params, params.filename, params.num_records)

    # test_ds = get_dataflow([FLAGS.filename])
    # itr = test_ds.as_numpy_iterator()
    # parsed_record = next(itr)
    # data_sample = transform_raw_record(parsed_record)
    # print(data_sample['actions'])


if __name__ == '__main__':
    main()


# nohup python -u supervised_data.py --filename=/data2/honerkam/pfnet_data/train/Rs0_floor0.tfrecord --scene_id=Rs --agent=avoid_agent --num_records=5 --custom_output rgb_obs depth_obs occupancy_grid obstacle_obs --config_file=./configs/locobot_pfnet_nav.yaml --env_mode=headless --device_idx=1 --seed=90 &> nohup1.out &
# nohup python -u supervised_data.py --filename=/data2/honerkam/pfnet_data/train_navagent/Rs0_floor0.tfrecord --scene_id=Rs --agent=goalnav_agent --num_records=7500 --custom_output rgb_obs depth_obs occupancy_grid obstacle_obs --config_file=./configs/locobot_pfnet_nav.yaml --env_mode=headless --device_idx=1 --seed=90 &> nohup_datacollection1.out &

#
# scenes=("Beechwood_0_int" "Benevolence_2_int" "Merom_1_int" "Rs_int" "Beechwood_1_int" "Ihlen_0_int" "Pomaria_0_int" "Wainscott_0_int" "Benevolence_0_int" "Ihlen_1_int" "Pomaria_1_int" "Wainscott_1_int" "Benevolence_1_int" "Merom_0_int" "Pomaria_2_int")
# # train_scenes=("Merom_0_int" "Benevolence_0_int" "Pomaria_0_int" "Wainscott_1_int" "Rs_int" "Ihlen_0_int" "Beechwood_1_int" "Ihlen_1_int")
# # test_scenes=("Benevolence_1_int" "Wainscott_0_int" "Pomaria_2_int" "Benevolence_2_int" "Beechwood_0_int" "Pomaria_1_int" "Merom_1_int")
#
# for scene in "${scenes[@]}"; do
#     nohup python -u supervised_data.py --filename=/data/honerkam/pfnet_data/navagent_allscenes/${scene}.tfrecord --scene_id=${scene} --agent=goalnav_agent --num_records=10000 --custom_output rgb_obs depth_obs occupancy_grid obstacle_obs --config_file=./configs/locobot_pfnet_nav.yaml --env_mode=headless --device_idx=3 --seed=90 &> nohup_datacollection${scene}.out &
# done


