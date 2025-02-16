#!/usr/bin/env python3

import argparse
import cv2
import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import os

from rl_agents.UNUSED import pfnet_loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
from scipy.special import softmax
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from pfnetwork import pfnet
from environments.env_utils import datautils, render
from environments.envs.localize_env import LocalizeGibsonEnv

np.set_printoptions(suppress=True)
def parse_args():
    """
    Parse command line arguments

    :return: argparse.Namespace
        parsed command-line arguments passed to *.py
    """

    # initialize parser
    arg_parser = argparse.ArgumentParser()

    # define training parameters
    arg_parser.add_argument(
        '--obs_mode',
        type=str,
        default='rgb-depth',
        help='Observation input type. Possible values: rgb / depth / rgb-depth / occupancy_grid.'
    )
    arg_parser.add_argument(
        '--custom_output',
        nargs='*',
        default=['rgb_obs', 'depth_obs', 'occupancy_grid', 'floor_map', 'kmeans_cluster', 'likelihood_map'],
        help='A comma-separated list of env observation types.'
    )
    arg_parser.add_argument(
        '--root_dir',
        type=str,
        default='./train_output',
        help='Root directory for logs/summaries/checkpoints.'
    )
    arg_parser.add_argument(
        '--tfrecordpath',
        type=str,
        default='./data',
        help='Folder path to training/evaluation/testing (tfrecord).'
    )
    arg_parser.add_argument(
        '--num_train_samples',
        type=int,
        default=1,
        help='Total number of samples to use for training. Total training samples will be num_train_samples=num_train_batches*batch_size'
    )
    arg_parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Minibatch size for training'
    )
    arg_parser.add_argument(
        '--pfnet_loadpath',
        type=str,
        default='',
        help='Load a previously trained pfnet model from a checkpoint file.'
    )
    arg_parser.add_argument(
        '--s_buffer_size',
        type=int,
        default=500,
        help='Buffer size for shuffling data'
    )
    arg_parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Fix the random seed'
    )
    arg_parser.add_argument(
        '--device_idx',
        type=int,
        default='0',
        help='use gpu no. to train/eval'
    )

    # define particle parameters
    arg_parser.add_argument(
        '--init_particles_distr',
        type=str,
        default='gaussian',
        help='Distribution of initial particles. Possible values: gaussian / uniform.'
    )
    arg_parser.add_argument(
        '--init_particles_std',
        nargs='*',
        default=["0.15", "0.523599"],
        help='Standard deviations for generated initial particles for tracking distribution.'
             'Values: translation std (meters), rotation std (radians)'
    )
    arg_parser.add_argument(
        '--particles_range',
        type=int,
        default=100,
        help='Pixel range to limit uniform distribution sampling +/- box particles_range center at robot pose'
    )
    arg_parser.add_argument(
        '--num_particles',
        type=int,
        default=30,
        help='Number of particles in Particle Filter'
    )
    arg_parser.add_argument(
        '--transition_std',
        nargs='*',
        default=["0.01", "0.0872665"],
        help='Standard deviations for transition model. Values: translation std (meters), rotation std (radians)'
    )
    arg_parser.add_argument(
        '--resample',
        type=str,
        default='false',
        help='Resample particles in Particle Filter'
    )
    arg_parser.add_argument(
        '--alpha_resample_ratio',
        type=float,
        default=1.0,
        help='Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true.'
             'Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling'
    )
    arg_parser.add_argument(
        '--global_map_size',
        nargs='*',
        default=["100", "100", "1"],
        help='Global map size in pixels (H, W, C)'
    )
    arg_parser.add_argument(
        '--window_scaler',
        type=float,
        default=1.0,
        help='Rescale factor for extracing local map'
    )

    # define igibson env parameters
    arg_parser.add_argument(
        '--config_file',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'configs',
            'turtlebot_pfnet_nav.yaml'
        ),
        help='Config file for the experiment'
    )
    arg_parser.add_argument(
        '--scene_id',
        type=str,
        default='Rs',
        help='Environment scene'
    )
    arg_parser.add_argument(
        '--action_timestep',
        type=float,
        default=1.0 / 10.0,
        help='Action time step for the simulator'
    )
    arg_parser.add_argument(
        '--physics_timestep',
        type=float,
        default=1.0 / 40.0,
        help='Physics time step for the simulator'
    )

    # parse parameters
    params = arg_parser.parse_args()

    # For the igibson maps, originally each pixel represents 0.01m, and the center of the image correspond to (0,0)
    params.map_pixel_in_meters = 0.01
    # in igibson we work with rescaled 0.01m to 0.1m maps to sample robot poses
    params.trav_map_resolution = 0.1

    # post-processing
    params.num_train_batches = params.num_train_samples//params.batch_size

    # convert multi-input fields to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)
    params.global_map_size = np.array(params.global_map_size, np.int32)

    params.transition_std[0] = (params.transition_std[0] / params.map_pixel_in_meters) * params.trav_map_resolution # convert meters to pixels and rescale to trav map resolution
    params.init_particles_std[0] = (params.init_particles_std[0] / params.map_pixel_in_meters) * params.trav_map_resolution  # convert meters to pixels and rescale to trav map resolution

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.init_particles_std.copy()
    particle_std2 = np.square(particle_std)  # variance
    params.init_particles_cov = np.diag(particle_std2[(0, 0, 1), ])

    if params.resample not in ['false', 'true']:
        raise ValueError
    else:
        params.resample = (params.resample == 'true')

    # use RNN as stateful/non-stateful
    params.stateful = False
    params.return_state = True

    # compute observation channel dim
    if params.obs_mode == 'rgb-depth':
        params.obs_ch = 4
    elif params.obs_mode == 'rgb':
        params.obs_ch = 3
    elif params.obs_mode == 'depth' or params.obs_mode == 'occupancy_grid':
        params.obs_ch = 1
    else:
        raise ValueError

    # HACK:
    params.loop = 6
    params.use_tf_function = False
    params.init_env_pfnet = True
    params.store_results = True
    params.num_clusters = 10
    params.use_plot = False
    params.store_plot = False

    params.env_mode = 'headless'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.device_idx)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # set random seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    return params

def display_data(arg_params):
    """
    """
    root_dir = os.path.expanduser(arg_params.root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    # training data
    filenames = list(glob.glob(os.path.join(arg_params.tfrecordpath, 'train', '*.tfrecord')))
    train_ds = datautils.get_dataflow(filenames, arg_params.batch_size, arg_params.s_buffer_size, is_training=True)
    print(f'train data: {filenames}')

    # create igibson env which is used "only" to sample particles
    env = LocalizeGibsonEnv(
        config_file=arg_params.config_file,
        scene_id=arg_params.scene_id,
        mode=arg_params.env_mode,
        use_tf_function=arg_params.use_tf_function,
        init_pfnet=arg_params.init_env_pfnet,
        action_timestep=arg_params.action_timestep,
        physics_timestep=arg_params.physics_timestep,
        device_idx=arg_params.device_idx,
        pf_params=arg_params
    )
    obs = env.reset()

    likelihood_map_ext = obs['likelihood_map']
    floor_map = env.floor_map[0].cpu().numpy()
    likelihood_map = np.zeros((*floor_map.shape[:2], 3))
    likelihood_map[:, :, :2] = likelihood_map_ext[:, :, :2]
    likelihood_map[:, :, 2] = np.arctan2(likelihood_map_ext[:, :, 3], likelihood_map_ext[:, :, 2])
    likelihood_map[:, :, 2] -= np.min(likelihood_map[:, :, 2])

    # normalize
    likelihood_map[:, :, 0] /= np.max(likelihood_map[:, :, 0])
    likelihood_map[:, :, 1] /= np.max(likelihood_map[:, :, 1])
    likelihood_map[:, :, 2] /= np.max(likelihood_map[:, :, 2])
    cv2.imwrite('./likelihood_map.png', cv2.flip(likelihood_map*255, 0))

    arg_params.trajlen = env.config.get('max_step', 500)//arg_params.loop
    arg_params.floors = 1
    assert arg_params.trav_map_resolution == env.trav_map_resolution
    print(arg_params)

    b_idx = 0
    t_idx = 10
    batch_size = arg_params.batch_size
    num_particles = arg_params.num_particles
    fig = plt.figure(figsize=(14, 14))
    plts = {}
    for idx in range(arg_params.floors):
        plts[idx] = fig.add_subplot(1,arg_params.floors,idx+1)

    # run training over all training samples in an epoch
    train_itr = train_ds.as_numpy_iterator()
    for idx in tqdm(range(arg_params.num_train_batches)):

        parsed_record = next(train_itr)
        batch_sample = datautils.transform_raw_record(env, parsed_record, arg_params)

        observation = batch_sample['observation'][b_idx]
        odometry = batch_sample['odometry'][b_idx]
        true_states = batch_sample['true_states'][b_idx]
        init_particles = batch_sample['init_particles'][b_idx]
        # init_particle_weights = np.full(shape=(batch_size, num_particles), fill_value=np.log(1.0 / float(num_particles)))[b_idx]
        init_particle_weights = np.random.random(size=(batch_size, num_particles))[b_idx]
        floor_map = batch_sample['floor_map'][b_idx]
        org_map_shape = batch_sample['org_map_shape'][b_idx]

        if arg_params.obs_mode == 'rgb-depth':
            rgb, depth = np.split(observation, [3], axis=-1)
            cv2.imwrite('./rgb.png', datautils.denormalize_observation(rgb)[t_idx])
            cv2.imwrite('./depth.png', cv2.applyColorMap(
                datautils.denormalize_observation(depth[t_idx]*255/100).astype(np.uint8),
                cv2.COLORMAP_JET))
        elif arg_params.obs_mode == 'depth':
            cv2.imwrite('./depth.png', cv2.applyColorMap(
                datautils.denormalize_observation(observation[t_idx]*255/100).astype(np.uint8),
                cv2.COLORMAP_JET))
        elif arg_params.obs_mode == 'rgb':
            cv2.imwrite('./rgb.png', datautils.denormalize_observation(observation[t_idx]))
        elif arg_params.obs_mode == 'occupancy_grid':
            cv2.imwrite('./occupancy_grid.png', observation[t_idx]*255)
        else:
            ValueError

        scene_id = parsed_record['scene_id'][b_idx][0].decode('utf-8')
        floor_num = parsed_record['floor_num'][b_idx][0]
        key = scene_id + '_' + str(floor_num)
        plt_ax = plts[floor_num]

        # floor map
        map_plt = render.draw_floor_map(floor_map, org_map_shape, plt_ax, None, cmap='gray')

        # init particles
        # HACK: display particles alpha proprtional to their weights
        init_lin_weights = softmax(init_particle_weights)
        th = np.mean(init_lin_weights)
        alphas = np.where(init_lin_weights >= th, 1, 0) * init_lin_weights
        alphas = alphas/np.max(alphas)

        part_col, part_row, part_th = np.split(init_particles, 3, axis=-1)
        rgba_colors = cm.rainbow(init_particle_weights-np.min(init_particle_weights))
        rgba_colors[:, 3] = alphas
        plt_ax.scatter(part_row, part_col, s=10, c=rgba_colors)

        # gt init pose
        c1, r1, th1 = true_states[0]
        heading_len  = robot_radius = 1.0
        xdata = [r1, r1 + (robot_radius + heading_len) * np.cos(th1)]
        ydata = [c1, c1 + (robot_radius + heading_len) * np.sin(th1)]
        position_plt = Wedge((r1, c1), r=robot_radius, theta1=0, theta2=360, color='blue', alpha=0.5)
        plt_ax.add_artist(position_plt)
        plt_ax.plot(xdata, ydata, color='blue', alpha=0.5)

        # # gt trajectory (w.r.t odometry)
        # c1, r1, th1 = true_states[0]
        # for t_idx in range(1, true_states.shape[0]):
        #     c2, r2, th2 = true_states[t_idx]
        #     plt_ax.arrow(r1, c1, (r2-r1), (c2-c1), head_width=0.5, head_length=0.7, fc='blue', ec='blue')
        #     c1, r1, th1 = c2, r2, th2

        # gt trajectory (w.r.t gt pose)
        c1, r1, th1 = true_states[0]
        for t_idx in range(0, odometry.shape[0]-1):
            c2, r2, th2 = datautils.sample_motion_odometry(np.array([c1, r1, th1]),odometry[t_idx])
            plt_ax.arrow(r1, c1, (r2-r1), (c2-c1), head_width=0.5, head_length=0.7, fc='black', ec='black')
            c1, r1, th1 = c2, r2, th2

    plt.tight_layout()
    for key, plt_ax in plts.items():
        extent = plt_ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{key}.png', bbox_inches=extent)
    fig.savefig('full_figure.png')

if __name__ == '__main__':
    parsed_params = parse_args()
    display_data(parsed_params)
