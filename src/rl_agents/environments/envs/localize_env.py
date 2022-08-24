#!/usr/bin/env python3

import argparse
import copy
import os
from collections import OrderedDict
from datetime import datetime

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from absl import flags
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.assets_utils import get_scene_path
from igibson.utils.utils import l2_distance
from matplotlib.backends.backend_agg import FigureCanvasAgg

# from igibson.external.pybullet_tools.utils import plan_base_motion_2d
from igibson.utils.utils import l2_distance, quatToXYZW, rotate_vector_2d, rotate_vector_3d

from ..env_utils import datautils, render

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from PIL import Image
from pfnetwork import pfnet
import pybullet as p
from sklearn.cluster import KMeans
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


ORIG_IGIBSON_MAP_RESOLUTION = 0.01


class LocalizeGibsonEnv(iGibsonEnv):
    """
    Custom implementation of localization task extending iGibsonEnv's functionality
    """

    def __init__(
            self,
            config_file,
            scene_id=None,
            mode='headless',
            # use_tf_function=True,
            pfnet_model=None,
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            device_idx=0,
            render_to_tensor=False,
            automatic_reset=False,
            pf_params=None,
    ):
        """
        Perform required iGibsonEnv initialization.
        In addition, override behaviour specific to localization task, which are:
        1. appropriate task reward function and task termination conditions
        2. initialize particle filter network
        3. initialize custom observation specs to return on env.step() and env.reset()

        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param use_tf_function: whether to wrap pfnetwork with tf.graph
        :param init_pfnet: whether to initialize pfnetwork
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param pf_params: argparse.Namespace parsed command-line arguments to initialize pfnet
        """

        super(LocalizeGibsonEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            automatic_reset=automatic_reset)

        # HACK: use termination_conditions: MaxCollision, Timeout, OutOfBound reward_functions: CollisionReward
        # manually remove geodesic potential reward conditions
        del self.task.reward_functions[0]

        # manually remove point navigation task termination and reward conditions
        del self.task.termination_conditions[-1]
        del self.task.reward_functions[-1]

        # For the igibson maps, originally each pixel represents 0.01m, and the center of the image correspond to (0,0)
        # self.map_pixel_in_meters = 0.01
        # in igibson we work with rescaled 0.01m to 0.1m maps to sample robot poses
        self.trav_map_resolution = self.config['trav_map_resolution']
        assert self.trav_map_resolution == pf_params.map_pixel_in_meters
        self.depth_th = 3.
        self.robot_size_px = 0.3 / self.trav_map_resolution  # 0.03m
        assert self.config['max_step'] // pf_params.loop == pf_params.trajlen, (self.config['max_step'], pf_params.trajlen)
        # assert pf_params.trav_map_resolution == self.trav_map_resolution, (pf_params.trav_map_resolution, self.trav_map_resolution)
        # argparser = argparse.ArgumentParser()
        # self.pf_params = argparser.parse_args([])
        # self.use_pfnet = init_pfnet
        # self.use_tf_function = use_tf_function
        # # initialize particle filter
        # if self.use_pfnet:
        #     print("=====> LocalizeGibsonEnv's pfnet initializing....")
        #     self.init_pfnet(pf_params)
        # else:
        #     self.pf_params.use_plot = False
        #     self.pf_params.store_plot = False
        #     if pf_params is not None:
        #         self.pf_params.num_clusters = pf_params.num_clusters
        #         self.pf_params.global_map_size = pf_params.global_map_size
        #         self.pf_params.custom_output = pf_params.custom_output
        #         self.pf_params.root_dir = pf_params.root_dir
        #     else:
        #         self.pf_params.num_clusters = 10
        #         self.pf_params.global_map_size = [100, 100, 1]
        #         self.pf_params.custom_output = ['rgb_obs', 'depth_obs', 'occupancy_grid', 'obstacle_obs']
        #         self.pf_params.root_dir = './'
        #         self.pf_params.loop = 6
        self.pfnet_model = pfnet_model
        assert pf_params is not None
        self.use_pfnet = self.pfnet_model is not None
        self.pf_params = pf_params
        if self.pf_params.use_plot:
            self.init_pfnet_plots()

        # for custom tf_agents we are using supports dict() type observations
        observation_space = OrderedDict()

        task_obs_dim = 7 # robot_prorpio_state (18)
        if 'task_obs' in self.pf_params.custom_output:
            # HACK: use [-1k, +1k] range for TanhNormalProjectionNetwork to work
            observation_space['task_obs'] = gym.spaces.Box(
                low=-1000.0, high=+1000.0,
                shape=(task_obs_dim,),
                dtype=np.float32)
        # image_height and image_width are obtained from env config file
        if 'rgb_obs' in self.pf_params.custom_output:
            observation_space['rgb_obs'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.float32)
        if 'depth_obs' in self.pf_params.custom_output:
            observation_space['depth_obs'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.image_height, self.image_width, 1),
                dtype=np.float32)
        if 'kmeans_cluster' in self.pf_params.custom_output:
            observation_space['kmeans_cluster'] = gym.spaces.Box(
                low=-1000.0, high=+1000.0,
                shape=(self.pf_params.num_clusters,5),
                dtype=np.float32)
        if 'raw_particles' in self.pf_params.custom_output:
            observation_space['raw_particles'] = gym.spaces.Box(
                low=-1000.0, high=+1000.0,
                shape=(self.pf_params.num_particles,5),
                dtype=np.float32)
        if 'floor_map' in self.pf_params.custom_output:
            observation_space['floor_map'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=self.pf_params.global_map_size,
                dtype=np.float32)
        if 'likelihood_map' in self.pf_params.custom_output:
            observation_space['likelihood_map'] = gym.spaces.Box(
                low=-10.0, high=+10.0,
                shape=(*self.pf_params.global_map_size[:2], 4),
                dtype=np.float32)
        if "occupancy_grid" in self.pf_params.custom_output:
            self.grid_resolution = self.config.get("grid_resolution", 128)
            observation_space['occupancy_grid'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.grid_resolution, self.grid_resolution, 1),
                dtype=np.float32)
        if "scan_obs" in self.pf_params.custom_output:
            observation_space["scan_obs"] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)

        print("=====> LocalizeGibsonEnv initialized")

        self.last_reward = 0.0



    @staticmethod
    def _get_empty_env_plots():
        return {
            'map_plt': None,
            'robot_gt_plt': {
                'position_plt': None,
                'heading_plt': None,
            },
            'robot_est_plt': {
                'position_plt': None,
                'heading_plt': None,
                'particles_plt': None,
            },
            'step_txt_plt': None,
        }

    def init_pfnet_plots(self):
        """
        Initialize Particle Filter

        :param pf_params: argparse.Namespace parsed command-line arguments to initialize pfnet
        """

        # initialize particle filter with input parameters otherwise use default values
        # if pf_params is not None:
        #     self.pf_params = pf_params
        # else:
        #     self.init_pf_params(flags.FLAGS)

        # # HACK: for real time particle filter update
        # self.pf_params.batch_size = 1
        # self.pf_params.trajlen = 1
        #
        # # Create a new pfnet model instance
        # self.pfnet_model = pfnet.pfnet_model(self.pf_params, is_igibson=True)
        # print(self.pf_params)
        # print("=====> LocalizeGibsonEnv's pfnet initialized")
        #
        # # load model from checkpoint file
        # if self.pf_params.pfnet_loadpath:
        #     # TODO: this won't work anymore, try the following for the future
        #     # tf.saved_model.save(my_model, "the_saved_model")
        #     # new_model = tf.saved_model.load("the_saved_model")
        #     # pass
        #     self.pfnet_model.load_weights(self.pf_params.pfnet_loadpath)
        #     print("=====> loaded pf model checkpoint " + self.pf_params.pfnet_loadpath)
        #
        # # wrap model with tf.graph
        # if self.use_tf_function:
        #     print("=====> wrapped pfnet in tf.graph")
        #     self.pfnet_model = tf.function(self.pfnet_model)

        # if self.pf_params.use_plot:
        # code related to displaying/storing results in matplotlib
        self.fig = plt.figure(figsize=(len(self.observation_space) * 6, 7))
        self.plt_ax = None
        self.env_plts = self._get_empty_env_plots()

        # HACK FigureCanvasAgg and ion is not working together
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out_folder = os.path.join(self.pf_params.root_dir, f'episode_run_{current_time}')
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)
        if self.pf_params.store_plot:
            self.canvas = FigureCanvasAgg(self.fig)
        else:
            plt.ion()
            plt.show()

    def _reset_vars(self):
        # self.obstacle_map = None
        self.floor_map = None

        self.eps_obs = {
            'rgb': [],
            'depth': [],
            'occupancy_grid': []
        }
        self.curr_plt_images = []
        self.curr_pfnet_state = None
        self.curr_obs = None
        self.curr_gt_pose = None
        self.curr_est_pose = None
        self.curr_cluster = None

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """

        super(LocalizeGibsonEnv, self).load_miscellaneous_variables()
        self._reset_vars()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """

        super(LocalizeGibsonEnv, self).reset_variables()
        self._reset_vars()

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions

        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """

        # HACK: we use low update frequency
        for _ in range(self.pf_params.loop):
            state, reward, done, info = super(LocalizeGibsonEnv, self).step(action)
        info['collision_penality'] = reward  # contains only collision reward per step

        # perform particle filter update
        if self.use_pfnet:
            # HACK: wrap stop_gradient to make sure pfnet weights are not updated during rl training
            loss_dict = self.step_pfnet(state)
            info['pred'] = loss_dict['pred']
            info['coords'] = loss_dict['coords']
            info['orient'] = loss_dict['orient']

            # include pfnet's estimate in environment's reward
            # TODO: may need better reward ?
            rescale = 10
            reward = (reward - tf.stop_gradient(loss_dict['pred'])) / rescale
            reward = tf.squeeze(reward)

        # just for the render function
        self.last_reward = reward

        # return custom environment observation
        custom_state = self.process_state(state)
        return custom_state, reward.cpu().numpy(), done, info

    def sample_initial_pose_and_target_pos(self):
        """
        Sample robot initial pose and target position
        :param env: environment instance
        :return: initial pose and target position
        """
        self.task.target_dist_min = self.task.config.get("target_dist_min", 1.0)
        self.task.target_dist_max = self.task.config.get("target_dist_max", 10.0)

        lmt = 1.0
        while True:
            _, initial_pos = self.scene.get_random_point(floor=self.task.floor_num)
            if -lmt <= initial_pos[0] <= lmt and -lmt <= initial_pos[1] <= lmt:
                break

        max_trials = 200
        dist = 0.0
        for _ in range(max_trials):
            _, target_pos = self.scene.get_random_point(floor=self.task.floor_num)
            if self.scene.build_graph:
                _, dist = self.scene.get_shortest_path(
                    self.task.floor_num, initial_pos[:2], target_pos[:2], entire_path=False
                )
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.task.target_dist_min < dist < self.task.target_dist_max:
                break
        if not (self.task.target_dist_min < dist < self.task.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn, target_pos

    def reset_agent(self):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.
        :param env: environment instance
        """
        reset_success = False
        max_trials = 200

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, target_pos = self.sample_initial_pose_and_target_pos()
            reset_success = self.test_valid_position(self.robots[0], initial_pos, initial_orn) and self.test_valid_position(self.robots[0], target_pos)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            print("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        self.task.target_pos = target_pos
        self.task.initial_pos = initial_pos
        self.task.initial_orn = initial_orn

    def reset(self):
        """
        Reset episode

        :return: state: new observation
        """

        if self.pf_params.use_plot:
            self.last_video_path = self.store_results()
            # self.store_obs()
            
            # clear subplots
            self.fig.clear()

            num_subplots = 5
            # for sensor in ['rgb_obs', 'depth_obs', 'occupancy_grid']:
            #     if self.observation_space.get(sensor, None):
            #         num_subplots += 1

            self.plt_ax = [self.fig.add_subplot(1, num_subplots, i + 1) for i in range(num_subplots)]
            self.plt_ax[0].set_title('iGibson Apartment')
            self.env_plts = self._get_empty_env_plots()

        # HACK: sample robot pose from selective area
        # self.reset_agent()

        state = super(LocalizeGibsonEnv, self).reset()

        # process new env map
        self.floor_map, self.org_map_shape, self.trav_map = self.get_floor_map(pad_map_size=self.pf_params.global_map_size)

        # perform particle filter update
        if self.use_pfnet:
            # get latest robot state
            new_robot_state = self.robots[0].calc_state()
            # process new robot state: convert coords to pixel space
            self.curr_gt_pose = self.get_robot_pose(new_robot_state, self.floor_map.shape)
            init_particles, init_particle_weights = pfnet.PFCell.reset(robot_pose_pixel=self.curr_gt_pose, env=self, params=self.pf_params)
            self.curr_est_pose = pfnet.PFCell.get_est_pose(particles=init_particles, particle_weights=init_particle_weights)

            self.curr_pfnet_state = [init_particles, init_particle_weights, self.floor_map[None]]

            if 'kmeans_cluster' in self.pf_params.custom_output:
                self.curr_cluster = self.compute_kmeans()

            # new_rgb_obs = copy.deepcopy(state['rgb'])  # [0, 1]
            # new_depth_obs = copy.deepcopy(state['depth'])  # [0, 1]
            # new_occupancy_grid = copy.deepcopy(state['occupancy_grid'])
            # pose_mse = self.reset_pfnet()['pred'].cpu().numpy()

        # return custom environment observation
        custom_state = self.process_state(state)
        return custom_state

    def process_state(self, state):
        """
        Perform additional processing of environment's observation.

        :param state: env observations

        :return: processed_state: processed env observations
        """
        assert np.min(state['rgb']) >= 0. and np.max(state['rgb']) <= 1., (np.min(state['rgb']), np.max(state['rgb']))
        assert np.min(state['depth']) >= 0. and np.max(state['depth']) <= 1., (np.min(state['depth']), np.max(state['depth']))

        self.eps_obs['rgb'].append((state['rgb'] * 255).astype(np.uint8))
        self.eps_obs['depth'].append(cv2.applyColorMap((state['depth'] * 255).astype(np.uint8), cv2.COLORMAP_JET))
        self.eps_obs['occupancy_grid'].append(cv2.cvtColor((state['occupancy_grid'] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

        # process and return only output we are expecting to
        processed_state = OrderedDict()
        if 'task_obs' in self.pf_params.custom_output:
            rpy = self.robots[0].get_rpy()
            # rotate linear and angular velocities to local frame
            lin_vel = rotate_vector_3d(self.robots[0].get_linear_velocity(), *rpy)
            ang_vel = rotate_vector_3d(self.robots[0].get_angular_velocity(), *rpy)
            # processed_state['task_obs'] = self.robots[0].calc_state()  # robot proprioceptive state
            processed_state['task_obs'] = np.concatenate([lin_vel, ang_vel, [self.collision_step]])
        if 'rgb_obs' in self.pf_params.custom_output:
            processed_state['rgb_obs'] = state['rgb']  # [0, 1] range rgb image
        if 'depth_obs' in self.pf_params.custom_output:
            processed_state['depth_obs'] = state['depth']  # [0, 1] range depth image
        if 'scan_obs' in self.pf_params.custom_output:
            processed_state['scan_obs'] = state['scan']
        if 'occupancy_grid' in self.pf_params.custom_output:
            # robot is at center facing right in grid
            processed_state['occupancy_grid'] = state['occupancy_grid'] #   [0: occupied, 0.5: unknown, 1: free]
        if 'kmeans_cluster' in self.pf_params.custom_output:
            if self.curr_cluster is not None:
                cluster_centers, cluster_weights = self.curr_cluster
                def cluster_pose(cluster_center, cluster_weight):
                    return np.array([*self.map_to_world(cluster_center[:2]), *cluster_center[2:], cluster_weight])   # cluster_pose [x, y, theta, weight] in mts

                processed_state['kmeans_cluster'] = np.stack([
                    np.append(cluster_centers[c_idx], cluster_weights[c_idx]) for c_idx in range(self.pf_params.num_clusters)   # cluster_pose [x, y, theta, weight] in px
                ])
            else:
                processed_state['kmeans_cluster'] = None
        if 'raw_particles' in self.pf_params.custom_output:
            if self.curr_pfnet_state is not None:
                particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
                lin_weights = tf.nn.softmax(particle_weights, axis=-1)[0].cpu().numpy()  # normalize weights
                particles = particles[0].cpu().numpy()

                assert list(particles.shape) == [self.pf_params.num_particles, 3]
                particles_ext = np.zeros((self.pf_params.num_particles, 4))
                particles_ext[:, :2] = particles[:, :2]
                particles_ext[:, 2] = np.cos(particles[:, 2])
                particles_ext[:, 3] = np.sin(particles[:, 2])
                processed_state['raw_particles'] = np.append(particles_ext, lin_weights)
            else:
                processed_state['raw_particles'] = None
        if 'floor_map' in self.pf_params.custom_output:
            if self.floor_map is None:
                floor_map, self.org_map_shape = self.get_floor_map(pad_map_size=self.pf_params.global_map_size)
                processed_state['floor_map'] = floor_map
            else:
                processed_state['floor_map'] = self.floor_map # [0, 2] range floor map
        if 'likelihood_map' in self.pf_params.custom_output:
            processed_state['likelihood_map'] = pfnet.PFCell.get_likelihood_map(particles=self.curr_pfnet_state[0],
                                                                                    particle_weights=self.curr_pfnet_state[1],
                                                                                    floor_map=self.floor_map)
        if 'obstacle_obs' in self.pf_params.custom_output:
            # check for close obstacles to robot
            min_depth = np.min(state['depth'] * 100, axis=0)
            s = min_depth.shape[0] // 4
            left = np.min(min_depth[:s]) < self.depth_th
            left_front = np.min(min_depth[s:2*s]) < self.depth_th
            right_front = np.min(min_depth[2*s:3*s]) < self.depth_th
            right = np.min(min_depth[3*s:]) < self.depth_th
            processed_state['obstacle_obs'] = np.array([left, left_front, right_front, right])

        return processed_state


    def step_pfnet(self, new_state):
        """
        Perform one particle filter update step

        :param new_obs: latest observation from env.step()

        :return loss_dict: dictionary of total loss and coordinate loss (in meters)
        """

        # trajlen = self.pf_params.trajlen
        batch_size = self.pf_params.batch_size
        num_particles = self.pf_params.num_particles
        obs_ch = self.pf_params.obs_ch
        obs_mode = self.pf_params.obs_mode

        # previous robot's pose, observation and particle filter state
        # old_rgb_obs, old_depth_obs, old_occupancy_grid = self.curr_obs
        # old_pfnet_state = self.curr_pfnet_state

        # get latest robot state
        new_robot_state = self.robots[0].calc_state()
        #
        # new_rgb_obs, new_depth_obs, new_occupancy_grid_orig = new_obs
        # # process new rgb observation: convert [0, 255] to [-1, +1] range

        def process_image(img, resize=None):
            if resize is not None:
                img = cv2.resize(img, resize)
            return np.atleast_3d(img.astype(np.float32))

        #
        # # process new depth observation: convert [0, 100] to [-1, +1] range
        # new_depth_obs = datautils.process_raw_image(new_depth_obs, resize=(56, 56))
        #
        # # process new occupancy_grid
        # if self.pf_params.likelihood_model == 'learned':
        #     new_occupancy_grid = datautils.decode_image(new_occupancy_grid_orig, resize=(56, 56)).astype(np.float32)
        # else:
        #     new_occupancy_grid = new_occupancy_grid_orig
        # new_occupancy_grid = np.atleast_3d(new_occupancy_grid)

        # process new robot state: convert coords to pixel space
        new_gt_pose = self.get_robot_pose(new_robot_state, self.floor_map.shape)

        # calculate actual odometry b/w old pose and new pose
        old_gt_pose = self.curr_gt_pose
        assert list(old_gt_pose.shape) == [3] and list(new_gt_pose.shape) == [3], f'{old_gt_pose.shape}, {new_gt_pose.shape}'
        odometry = datautils.calc_odometry(old_gt_pose, new_gt_pose)

        # # convert to tensor, add batch_dim
        # new_rgb_obs = tf.expand_dims(tf.convert_to_tensor(new_rgb_obs, dtype=tf.float32), axis=0)
        # new_depth_obs = tf.expand_dims(tf.convert_to_tensor(new_depth_obs, dtype=tf.float32), axis=0)
        # new_occupancy_grid = tf.expand_dims(tf.convert_to_tensor(new_occupancy_grid, dtype=tf.float32), axis=0)
        # new_odom = tf.expand_dims(tf.convert_to_tensor(new_odom, dtype=tf.float32), axis=0)
        # new_pose = tf.expand_dims(tf.convert_to_tensor(new_pose, dtype=tf.float32), axis=0)
        # odometry = new_odom

        # add traj_dim
        new_depth_obs = process_image(new_state['depth'], resize=(56, 56))
        new_rgb_obs = process_image(new_state['rgb'], resize=(56, 56))
        if obs_mode == 'rgb-depth':
            observation = tf.concat([new_rgb_obs, new_depth_obs], axis=-1)
        elif obs_mode == 'depth':
            observation = new_depth_obs
        elif obs_mode == 'rgb':
            observation = new_rgb_obs
        elif obs_mode == 'occupancy_grid':
            if self.pf_params.likelihood_model == 'learned':
                observation = process_image(new_state['occupancy_grid'], resize=(56, 56))
            else:
                observation = process_image(new_state['occupancy_grid'], resize=None)
        else:
            raise ValueError(obs_mode)

        # sanity check
        def _add_batch_dim(x, add_traj_dim: bool = False):
            x = tf.expand_dims(tf.convert_to_tensor(x, dtype=tf.float32), axis=0)
            if add_traj_dim:
                x = tf.expand_dims(x, 1)
            return x
        odometry = _add_batch_dim(odometry, add_traj_dim=True)
        observation = _add_batch_dim(observation, add_traj_dim=True)
        # floor_map = _add_batch_dim(self.floor_map)

        old_pfnet_state = self.curr_pfnet_state
        trajlen = 1
        assert list(odometry.shape) == [batch_size, trajlen, 3], f'{odometry.shape}'
        assert list(observation.shape) in [[batch_size, trajlen, 56, 56, obs_ch], [batch_size, trajlen, 128, 128, obs_ch]], f'{observation.shape}'
        assert list(old_pfnet_state[0].shape) == [batch_size, num_particles, 3], f'{old_pfnet_state[0].shape}'
        assert list(old_pfnet_state[1].shape) == [batch_size, num_particles], f'{old_pfnet_state[1].shape}'
        assert list(old_pfnet_state[2].shape) == [batch_size] + list(self.floor_map.shape), f'{old_pfnet_state[2].shape}'

        # forward pass pfnet (in eval mode)
        # output: contains particles and weights before transition update
        # pfnet_state: contains particles and weights after transition update
        # output, new_pfnet_state = self.pfnet_model(tf.stop_gradient(model_input))
        # TODO: don't calculate gradients on anything here
        # self.curr_est_pose = tf.stop_gradient(self.pfnet_model(global_map=floor_map, odometry=odometry, observation=observation))

        curr_input = [observation, odometry]
        output, new_pfnet_state = self.pfnet_model((curr_input, old_pfnet_state), training=False)
        particles, particle_weights = output
        self.curr_est_pose = tf.stop_gradient(pfnet.PFCell.get_est_pose(particles=particles, particle_weights=particle_weights))

        # TODO: remove test
        # from pfnetwork.pfnet import calc_scan_correlation
        # p_states, p_weights, g_map = new_pfnet_state
        # o = tf.expand_dims(tf.convert_to_tensor(np.atleast_3d(new_occupancy_grid_orig), dtype=tf.float32), axis=0)
        # lik = calc_scan_correlation(g_map, p_states, o, window_scaler=self.pf_params.window_scaler, scan_size=o.shape[1])


        # compute pfnet loss, add traj_dim
        # particles, particle_weights = output
        # true_old_pose = tf.expand_dims(self.curr_gt_pose, axis=1)
        # particles = tf.expand_dims(particles, axis=1)
        # particle_weights = tf.expand_dims(particle_weights, axis=1)

        # sanity check
        # assert list(true_old_pose.shape) == [batch_size, trajlen, 3], f'{true_old_pose.shape}'
        # assert list(particles.shape) == [batch_size, trajlen, num_particles, 3], f'{particles.shape}'
        # assert list(particle_weights.shape) == [batch_size, trajlen, num_particles], f'{particle_weights.shape}'
        loss_dict = pfnet.PFCell.compute_mse_loss(particles=particles,
                                                  particle_weights=particle_weights,
                                                  true_states=_add_batch_dim(new_gt_pose),
                                                  trav_map_resolution=self.trav_map_resolution)

        # latest robot's pose, observation and particle filter state
        self.curr_pfnet_state = new_pfnet_state
        self.curr_gt_pose = new_gt_pose
        # self.curr_est_pose = self.get_est_pose()
        # self.curr_obs = [
        #     new_rgb_obs,
        #     new_depth_obs,
        #     new_occupancy_grid
        # ]
        if 'kmeans_cluster' in self.pf_params.custom_output:
            self.curr_cluster = self.compute_kmeans()

        return loss_dict


    def compute_kmeans(self):
        """
        Construct KMeans of particles & particle_weights from particle filter's current state

        :return cluster_centers: input 'num_clusters' computed cluster centers
        :return cluster_weights: corresponding particles in cluster total particle_weights
        """

        num_clusters = self.pf_params.num_clusters
        particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)[0].cpu().numpy()
        particles = particles[0].cpu().numpy()

        # expand orientation to corresponding cos and sine components
        assert list(particles.shape) == [self.pf_params.num_particles, 3]
        particles_ext = np.zeros((self.pf_params.num_particles, 4))
        particles_ext[:, :2] = particles[:, :2]
        particles_ext[:, 2] = np.cos(particles[:, 2])
        particles_ext[:, 3] = np.sin(particles[:, 2])

        if self.curr_cluster is None:
            # random initialization
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        else:
            # previous cluster center as initialization guess
            prev_cluster_centers, _ = self.curr_cluster
            assert list(prev_cluster_centers.shape) == [num_clusters, 4]
            kmeans = KMeans(n_clusters=num_clusters, init=prev_cluster_centers, n_init=1)
        kmeans.fit_predict(particles_ext)
        cluster_indices = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        assert list(lin_weights.shape) == list(cluster_indices.shape)
        cluster_weights = np.array([
                            np.sum(lin_weights[cluster_indices==c_idx]) for c_idx in range(num_clusters)
                        ])

        return cluster_centers, cluster_weights

    def set_scene(self, scene_id, floor_num):
        """
        Override the task floor number

        :param str: scene id
        :param int: task floor number
        """
        self.config['scene_id'] = scene_id
        self.task.floor_num = floor_num


    # def get_obstacle_map(self, scene_id=None, floor_num=None, pad_map_size=None):
    #     """
    #     Get the scene obstacle map
    #
    #     :param str: scene id
    #     :param int: task floor number
    #     :return ndarray: obstacle map of current scene (H, W, 1)
    #     """
    #     if scene_id is not None and floor_num is not None:
    #         self.set_scene(scene_id, floor_num)
    #
    #     obstacle_map = np.array(Image.open(
    #         os.path.join(get_scene_path(self.config.get('scene_id')),
    #                      f'floor_{self.task.floor_num}.png')
    #     ))
    #
    #     # HACK: use same rescaling as in iGibsonEnv
    #     height, width = obstacle_map.shape
    #     resize = int(height * self.map_pixel_in_meters / self.trav_map_resolution)
    #     obstacle_map = cv2.resize(obstacle_map, (resize, resize))
    #
    #     # process new obstacle map: convert [0, 255] to [0, 2] range
    #     obstacle_map = datautils.process_raw_map(obstacle_map)
    #     org_map_shape = obstacle_map.shape
    #
    #     # HACK: right zero-pad floor/obstacle map
    #     if pad_map_size is not None:
    #         obstacle_map = datautils.pad_images(obstacle_map, pad_map_size)
    #
    #     return obstacle_map, org_map_shape


    def get_floor_map(self, scene_id=None, floor_num=None, pad_map_size=None):
        """
        Get the scene floor map (traversability map + obstacle map)

        :param str: scene id
        :param int: task floor number
        :return ndarray: floor map of current scene (H, W, 1)
        """
        if scene_id is not None and floor_num is not None:
            self.set_scene(scene_id, floor_num)

        obstacle_map = np.array(Image.open(
            os.path.join(get_scene_path(self.config.get('scene_id')), f'floor_{self.task.floor_num}.png'))
        )

        trav_map = np.array(Image.open(
            os.path.join(get_scene_path(self.config.get('scene_id')), f'floor_trav_{self.task.floor_num}.png'))
        )

        # remove unnecessary empty map parts
        # bb = pfnet.PFCell.bounding_box(obstacle_map == 0)
        # obstacle_map = obstacle_map[bb[0]: bb[1], bb[2]: bb[3]]
        # trav_map = trav_map[bb[0]: bb[1], bb[2]: bb[3]]

        # 0: free / unexplored / outside map, 1: obstacle
        # NOTE: shouldn't mather that outside the map is not unexplored, because this will always be unexplored in the lidar scan and so will always be masked out
        occupancy_map = np.zeros_like(trav_map)
        # occupancy_map[trav_map == 0] = 2
        occupancy_map[obstacle_map == 0] = 1

        height, width = trav_map.shape
        resize = (int(width * ORIG_IGIBSON_MAP_RESOLUTION / self.trav_map_resolution),
                  int(height * ORIG_IGIBSON_MAP_RESOLUTION / self.trav_map_resolution))
        occupancy_map_small = cv2.resize(occupancy_map.astype(float), resize, interpolation=cv2.INTER_AREA)
        occupancy_map_small = occupancy_map_small[:, :, np.newaxis]
        # plt.imshow(occupancy_map_small > 0.1);
        # plt.show()

        # o = obstacle_map.copy()
        # flood_fill_flags = 4
        # h, w = o.shape[:2]
        # flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        # cv2.floodFill(o, mask=None, seedPoint=(h//2, w//2), newVal=125)
        # plt.imshow(o); plt.show()

        trav_map[obstacle_map == 0] = 0

        # HACK: use same rescaling as in iGibsonEnv
        # height, width = trav_map.shape
        # resize = int(height * self.map_pixel_in_meters / self.trav_map_resolution)
        trav_map = cv2.resize(trav_map, resize)
        trav_map_erosion = self.config.get('trav_map_erosion', 2)
        trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))
        # 1: traversible
        trav_map[trav_map < 255] = 0
        trav_map[trav_map == 255] = 1
        trav_map = trav_map[:, :, np.newaxis]

        self.trav_map_size = np.array(trav_map.shape[:2])

        # HACK: right zero-pad floor/obstacle map
        if pad_map_size is not None:
            trav_map = datautils.pad_images(trav_map, pad_map_size)
            occupancy_map_small = datautils.pad_images(occupancy_map_small, pad_map_size)

        return occupancy_map_small, occupancy_map_small.shape, trav_map

    def map_to_world(self, xy):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        :param xy: 2D location in map reference frame (image)
        :return: 2D location in world reference frame (metric)
        """
        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - self.trav_map_size / 2.0) * self.trav_map_resolution, axis=axis)

    def world_to_map(self, xy):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

        :param xy: 2D location in world reference frame (metric)
        :return: 2D location in map reference frame (image)
        """
        return np.flip((np.array(xy) / self.trav_map_resolution + self.trav_map_size / 2.0)).astype(int)

    def get_random_points_map(self, npoints, true_mask = None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.

        :param floor: floor number
        :return floor: floor number
        :return point: randomly sampled point in [x, y, z]
        """
        trav_map = self.trav_map.copy()
        if true_mask is not None:
            trav_map *= true_mask

        trav_space = np.where(trav_map == 1)
        idx = np.random.randint(0, high=trav_space[0].shape[0], size=npoints)
        orn = np.random.uniform(0, np.pi * 2, size=npoints)

        xy_map = np.stack([trav_space[0][idx], trav_space[1][idx], orn], 1)

        # xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        # x, y = self.scene.map_to_world(xy_map)
        # return np.array([x, y])
        return xy_map

    def get_robot_pose(self, robot_state, floor_map_shape):
        """
        Transform robot's pose from coordinate space to pixel space.
        """
        robot_pos = robot_state[0:3]  # [x, y, z]
        robot_orn = robot_state[3:6]  # [r, p, y]

        # transform from co-ordinate space [x, y] to pixel space [col, row]
        robot_pose_px = np.array([*self.world_to_map(robot_pos[:2]), robot_orn[2]])  # [x, y, theta]

        return robot_pose_px

    def plot_robot_pose(self, pose=None, floor_map=None):
        if pose is None:
            pose = self.curr_gt_pose
        if floor_map is None:
            floor_map = self.floor_map
        f, ax = plt.subplots(1, 1)
        render.draw_floor_map(floor_map, self.org_map_shape, ax, None)
        render.draw_robot_pose(pose, '#7B241C', floor_map.shape, ax, None, None, plt_path=True)
        return f, ax

    def render(self, mode='human', particles=None, particle_weights=None, floor_map=None, observation=None, gt_pose=None, current_step=None, curr_cluster=None, est_pose=None):
        """
        Render plots
        """
        # super(LocalizeGibsonEnv, self).render(mode)

        if self.pf_params.use_plot:
            if particles is None:
                if self.curr_pfnet_state is not None:
                    particles, particle_weights, floor_map = self.curr_pfnet_state
                    floor_map = np.squeeze(floor_map, 0)
                else:
                    floor_map = self.floor_map
                if gt_pose is None:
                    gt_pose = self.curr_gt_pose
                if est_pose is None:
                    est_pose = np.squeeze(self.curr_est_pose) if (self.curr_est_pose is not None) else None
                curr_cluster = self.curr_cluster
                current_step = self.current_step
            else:
                est_pose = np.squeeze(pfnet.PFCell.get_est_pose(particles=particles, particle_weights=particle_weights))

            if self.use_pfnet and ("likelihood_map" in self.pf_params.custom_output):
                # # belief map / likelihood map
                likelihood_map = pfnet.PFCell.get_likelihood_map(particles=particles,
                                                                 particle_weights=particle_weights,
                                                                 floor_map=floor_map)
                # likelihood_map = np.zeros((*floor_map.shape[:2], 3))
                # likelihood_map[:, :, :2] = likelihood_map_ext[:, :, :2]
                # likelihood_map[:, :, 2] = np.arctan2(likelihood_map_ext[:, :, 3], likelihood_map_ext[:, :, 2])
                # likelihood_map[:, :, 2] -= np.min(likelihood_map[:, :, 2])
                #
                #
                # # convert to [0, 1] range
                # likelihood_map[:, :, 0] /= np.max(likelihood_map[:, :, 0])
                # likelihood_map[:, :, 1] /= np.max(likelihood_map[:, :, 1])
                # likelihood_map[:, :, 2] /= np.max(likelihood_map[:, :, 2])

                map_plt = self.env_plts['map_plt']
                self.env_plts['map_plt'] = render.draw_floor_map(0.025 * likelihood_map[..., 0] + likelihood_map[..., 1], self.org_map_shape, self.plt_ax[0], map_plt, cmap=None)
            else:
                # environment map
                map_plt = self.env_plts['map_plt']
                self.env_plts['map_plt'] = render.draw_floor_map(floor_map, self.org_map_shape, self.plt_ax[0], map_plt)

            # ground truth robot pose and heading
            color = '#7B241C'
            # gt_pose = self.curr_gt_pose
            position_plt = self.env_plts['robot_gt_plt']['position_plt']
            heading_plt = self.env_plts['robot_gt_plt']['heading_plt']
            position_plt, heading_plt = render.draw_robot_pose(
                gt_pose,
                color,
                floor_map.shape,
                self.plt_ax[0],
                position_plt,
                heading_plt,
                plt_path=True)
            self.env_plts['robot_gt_plt']['position_plt'] = position_plt
            self.env_plts['robot_gt_plt']['heading_plt'] = heading_plt


            # estimated robot pose and heading
            if est_pose is not None:
                color = '#515A5A'
                # est_pose = np.squeeze(self.curr_est_pose[0].cpu().numpy())
                position_plt = self.env_plts['robot_est_plt']['position_plt']
                heading_plt = self.env_plts['robot_est_plt']['heading_plt']
                position_plt, heading_plt = render.draw_robot_pose(
                    est_pose,
                    color,
                    floor_map.shape,
                    self.plt_ax[0],
                    position_plt,
                    heading_plt,
                    plt_path=False)
                self.env_plts['robot_est_plt']['position_plt'] = position_plt
                self.env_plts['robot_est_plt']['heading_plt'] = heading_plt

            if "raw_particles" in self.pf_params.custom_output:
                # raw particles color coded using weights
                # particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
                lin_weights = tf.nn.softmax(particle_weights, axis=-1)
                particles_plt = self.env_plts['robot_est_plt']['particles_plt']
                particles_plt = render.draw_particles_pose(
                    particles[0].cpu().numpy(),
                    lin_weights[0].cpu().numpy(),
                    floor_map.shape,
                    particles_plt)
                self.env_plts['robot_est_plt']['particles_plt'] = particles_plt
            elif "kmeans_cluster" in self.pf_params.custom_output:
                # kmeans-cluster particles color coded using weights
                cc_particles_ext, cc_weights = curr_cluster
                cc_particles = np.zeros((self.pf_params.num_clusters, 3))
                cc_particles[:, :2] = cc_particles_ext[:, :2]
                cc_particles[:, 2] = np.arctan2(cc_particles_ext[:, 3], cc_particles_ext[:, 2])

                particles_plt = self.env_plts['robot_est_plt']['particles_plt']
                particles_plt = render.draw_particles_pose(
                    cc_particles,
                    cc_weights,
                    floor_map.shape,
                    particles_plt)
                self.env_plts['robot_est_plt']['particles_plt'] = particles_plt

            # # episode info
            # step_txt_plt = self.env_plts['step_txt_plt']
            # step_txt_plt = render.draw_text(
            #     f'episode: {self.current_episode}, step: {self.current_step}',
            #     '#7B241C', self.plt_ax, step_txt_plt)
            # self.env_plts['step_txt_plt'] = step_txt_plt

            # pose mse in mts
            if (est_pose is not None) and (gt_pose is not None):
                gt_pose_mts = np.array([*self.map_to_world(gt_pose[:2]), gt_pose[2]])
                est_pose_mts = np.array([*self.map_to_world(est_pose[:2]), est_pose[2]])
                pose_diff = gt_pose_mts - est_pose_mts
                pose_diff[-1] = datautils.normalize(pose_diff[-1]) # normalize
                pose_error = np.linalg.norm(pose_diff[..., :2])
            else:
                pose_error = 0.
            has_collision = ' True' if len(self.collision_links) > 0 else 'False'

            step_txt_plt = self.env_plts['step_txt_plt']
            # step_txt_plt = render.draw_text(
            #     f' pose mse: {np.linalg.norm(pose_diff):02.3f}\n collisions: {self.collision_step:03.0f}/{self.current_step:03.0f}',
            #     '#7B241C', self.plt_ax, step_txt_plt)
            step_txt_plt = render.draw_text(
                f' pose mse: {pose_error:02.3f}\n current step: {current_step//self.pf_params.loop:02.0f}\n last reward: {np.squeeze(self.last_reward):02.3f}\n collision: {has_collision}',
                '#FFFFFF', self.plt_ax[0], step_txt_plt)
            self.env_plts['step_txt_plt'] = step_txt_plt
            # print(f'gt_pose: {gt_pose_mts}, est_pose: {est_pose_mts} in mts')

            self.plt_ax[0].legend([self.env_plts['robot_gt_plt']['position_plt'],
                                self.env_plts['robot_est_plt']['position_plt']],
                               ["GT Pose", "Est Pose"], loc='upper left', fontsize=12)

            # plot the local map extracted for the ground-truth pose
            local_map = pfnet.PFCell.transform_maps(global_map=tf.convert_to_tensor(floor_map[None], tf.float32),
                                                    particle_states=tf.convert_to_tensor(gt_pose, tf.float32)[None, None],
                                                    local_map_size=(28, 28),
                                                    window_scaler=self.pf_params.window_scaler,
                                                    agent_at_bottom=True,
                                                    flip_map=True)
            self.plt_ax[1].imshow(np.squeeze(local_map))

            next_subplot = 2
            if observation is not None:
                if self.pf_params.obs_mode == "rgb-depth":
                    self.plt_ax[next_subplot + 1].imshow(np.squeeze(observation[..., 3]))
                    observation = observation[..., :3]
                self.plt_ax[next_subplot].imshow(np.squeeze(observation))

            else:
                if self.eps_obs.get('rgb', None):
                    self.plt_ax[next_subplot].imshow(self.eps_obs['rgb'][-1])
                    next_subplot += 1
                if self.eps_obs.get('depth', None):
                    self.plt_ax[next_subplot].imshow(self.eps_obs['depth'][-1])
                    next_subplot += 1
                if self.eps_obs.get('occupancy_grid', None):
                    self.plt_ax[next_subplot].imshow(self.eps_obs['occupancy_grid'][-1])
                    next_subplot += 1


            if self.pf_params.store_plot:
                self.canvas.draw()
                plt_img = np.array(self.canvas.renderer._renderer)
                plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)
                self.curr_plt_images.append(plt_img)
            else:
                plt.draw()
                plt.pause(0.00000000001)
                
                f = self.fig

                self.fig.show()

    def close(self):
        """
        environment close()
        """
        super(LocalizeGibsonEnv, self).close()

        # store the plots as video
        if self.pf_params.use_plot:
            if self.pf_params.store_plot:
                self.store_results()
                # self.store_obs()
            else:
                # to prevent plot from closing after environment is closed
                plt.ioff()
                plt.show()

        print("=====> iGibsonEnv closed")

    @staticmethod
    def convert_imgs_to_video(images, file_path):
        """
        Convert images to video
        """
        fps = 5
        frame_size = (images[0].shape[1], images[0].shape[0])
        out = cv2.VideoWriter(file_path,
                              cv2.VideoWriter_fourcc(*'MP4V'),
                              fps, 
                              frame_size)
        for img in images:
            out.write(img)
        out.release()

    def store_obs(self):
        """
        Store the episode environment's observations as video
        """
        for m in ['rgb', 'depth', 'occupancy_grid']:
            if len(self.eps_obs[m]) > 1:
                file_path = os.path.join(self.out_folder, f'{m}_episode_run_{self.current_episode}.mp4')
                self.convert_imgs_to_video(self.eps_obs[m], file_path)
                print(f'stored {m} imgs {len(self.eps_obs[m])} to {file_path}')
                self.eps_obs[m] = []

    def store_results(self):
        """
        Store the episode environment's belief map/likelihood map as video
        """
        if len(self.curr_plt_images) > 0:
            file_path = os.path.join(self.out_folder, f'episode_run_{self.current_episode}.mp4')
            self.convert_imgs_to_video(self.curr_plt_images, file_path)
            print(f'stored img results {len(self.curr_plt_images)} eps steps to {file_path}')
            self.curr_plt_images = []
            return file_path
        else:
            print('no plots available to store, check if env.render() is being called')


    def __del__(self):
        if len(self.curr_plt_images) > 0:
            self.close()
