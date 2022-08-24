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

    print(params)
    collect_data(env, params, params.filename, params.num_records)

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


