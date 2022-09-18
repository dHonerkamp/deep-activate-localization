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
from pathlib import Path

# import custom tf_agents
from environments.env_utils import datautils
from environments.envs.localize_env import LocalizeGibsonEnv
from pfnetwork.arguments import parse_common_args
from pfnetwork import pfnet


def collect_data(env, params, filename, num_records=10):
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
    try:
        ds = datautils.get_dataflow([filename], batch_size=1, s_buffer_size=100, is_training=False, is_igibson=True)
        data_itr = ds.as_numpy_iterator()
        for idx in range(num_records):
            parsed_record = next(data_itr)
            batch_sample = datautils.transform_raw_record(parsed_record, params)
    except BaseException as e:
        print(f"Sanity check for {env.config['scene_id']} failed with {e}")
        return
        

def collect_scene(params, scene_id: str, dir: str):
    env = LocalizeGibsonEnv(
        config_file=params.config_file,
        scene_id=scene_id,
        mode=params.env_mode,
        # use_tf_function=False,
        pfnet_model=None,
        pf_params=params,
        action_timestep=params.action_timestep,
        physics_timestep=params.physics_timestep,
        device_idx=params.device_idx
    )

    filename = f'{dir}/{scene_id}_floor{env.task.floor_num}.tfrecord'
    if os.path.exists(filename):
        print(f'File {os.path.abspath(filename)} already exists !!!')
        return
            
    collect_data(env, params, filename, params.num_records)
    
    
def get_scene_sizes():
    from PIL import Image
    import matplotlib.pyplot as plt
    # d = Path('/home/honerkam/repos/iGibson/igibson/data/g_dataset/')
    # parent folder of this repository as the data is in the iGibson repo
    d = Path(__file__).parent.parent.parent.parent / "iGibson/igibson/data/data_100scenes/"
    scene_paths = d.iterdir()
    sizes, scene_ids = [], []
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
        scene_ids.append(scene_path.name)
        print(scene_path.name, '\t', img.size, '\t', bb_size)
    sizes = np.stack(sizes)
    return sizes, scene_ids


def get_small_scenes(global_map_size):
    sizes, scene_ids = get_scene_sizes()
    
    scenes = []
    for s, scene_id in zip(sizes, scene_ids):
        if np.max(s) <= np.max(global_map_size):
            scenes.append(scene_id)  
    return scenes


def get_scene_ids(global_map_size):
    p = f"/data/honerkam/pfnet_data/train_navagent_below{np.max(global_map_size)}"
    if not Path(p).exists():
        p = p.replace("data", "data2")
    if Path(p).exists():
        train_scenes = [f.name.split("_")[0] for f in Path(p).glob("*.tfrecord")]
        test_scenes = [f.name.split("_")[0] for f in Path(p.replace("train", "test")).glob("*.tfrecord")]
    else:
        scene_ids = get_small_scenes(global_map_size)
        scene_ids = np.random.permutation(scene_ids)
        
        split = int((1 - 0.15) * len(scene_ids))
        train_scenes, test_scenes = scene_ids[:split], scene_ids[split:]
        print(f"{len(train_scenes)} train scenes, {len(test_scenes)} test scenes")
    return list(train_scenes), list(test_scenes)
    

def collect_all_scenes(params):
    import multiprocessing
    multiprocessing.set_start_method('forkserver')
    max_processes = 16
    
    train_scenes, test_scenes = get_scene_ids(params.global_map_size)
    
    dir = Path("/data/honerkam/pfnet_data")
    train_dir = dir / f"train_navagent_below{np.max(params.global_map_size)}"
    test_dir = dir / f"test_navagent_below{np.max(params.global_map_size)}"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    with multiprocessing.Pool(processes=max_processes) as pool:
        out = pool.starmap(collect_scene, 
                           [(params, scene, train_dir if (scene in train_scenes) else test_dir) for scene in train_scenes + test_scenes],
                           chunksize=1)
                
    print("done", out)
    
    
def main():
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()

    params = parse_common_args(env_name='igibson', collect_data=True)
    
    if params.debug:
        dir = Path(params.filename).absolute()
        dir.mkdir(exist_ok=True)
        f = Path("/data/honerkam/pfnet_data/blub/Rs_floor0.tfrecord") 
        if f.exists():
            f.unlink()
            
        collect_scene(params, dir=dir, scene_id=params.scene_id)
    else:
        collect_all_scenes(params)


if __name__ == '__main__':
    main()


# nohup python -u supervised_data.py --filename=/data2/honerkam/pfnet_data/train/Rs0_floor0.tfrecord --scene_id=Rs --agent=avoid_agent --num_records=5 --custom_output rgb_obs depth_obs occupancy_grid obstacle_obs  --device_idx=1 --seed=90 &> nohup1.out &
# nohup python -u supervised_data.py --filename=/data2/honerkam/pfnet_data/train_navagent/Rs0_floor0.tfrecord --scene_id=Rs --agent=goalnav_agent --num_records=7500 --custom_output rgb_obs depth_obs occupancy_grid obstacle_obs  --device_idx=1 --seed=90 &> nohup_datacollection1.out &
# nohup python -u supervised_data.py --device_idx 2 --agent "goalnav_agent" --num_records "1000" --custom_output "rgb_obs" "depth_obs" "occupancy_grid" "task_obs" "obstacle_obs"  --global_map_size "1000" "1000" "1" &> nohup_datacollection1.out &
#
# scenes=("Beechwood_0_int" "Benevolence_2_int" "Merom_1_int" "Rs_int" "Beechwood_1_int" "Ihlen_0_int" "Pomaria_0_int" "Wainscott_0_int" "Benevolence_0_int" "Ihlen_1_int" "Pomaria_1_int" "Wainscott_1_int" "Benevolence_1_int" "Merom_0_int" "Pomaria_2_int")
# # train_scenes=("Merom_0_int" "Benevolence_0_int" "Pomaria_0_int" "Wainscott_1_int" "Rs_int" "Ihlen_0_int" "Beechwood_1_int" "Ihlen_1_int")
# # test_scenes=("Benevolence_1_int" "Wainscott_0_int" "Pomaria_2_int" "Benevolence_2_int" "Beechwood_0_int" "Pomaria_1_int" "Merom_1_int")
#
# for scene in "${scenes[@]}"; do
#     nohup python -u supervised_data.py --filename=/data/honerkam/pfnet_data/navagent_allscenes/${scene}.tfrecord --scene_id=${scene} --agent=goalnav_agent --num_records=10000 --custom_output rgb_obs depth_obs occupancy_grid obstacle_obs --device_idx=3 --seed=90 &> nohup_datacollection${scene}.out &
# done


