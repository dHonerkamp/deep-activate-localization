#!/usr/bin/env python3

import argparse
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import cv2
import pybullet as p

try:
    from .architecture import networks
    from .architecture.spatial_transformer import transformer
except:
    from architecture import networks
    from architecture.spatial_transformer import transformer

from environments.env_utils.datautils import get_random_particles, get_random_points_map


def cosine_similarity(a, b, dim=-1):
    raise NotImplementedError("Untested, might still have bugs")
    if isinstance(a, tf.Tensor):
        return tf.inner(a, b, dim) / (tf.linalg.norm(a, dim) * tf.linalg.norm(b, dim))
    else:
        return np.inner(a, b, dim) / (np.linalg.norm(a, dim) * np.linalg.norm(b, dim))


def datautils_norm_angle(angle):
    """
    Normalize the angle to [-pi, pi]
    :param float angle: input angle to be normalized
    :return float: normalized angle
    """
    quaternion = p.getQuaternionFromEuler(np.array([0, 0, angle]))
    euler = p.getEulerFromQuaternion(quaternion)
    return euler[2]

def norm_angle(angle):
    return tf.math.floormod(angle + np.pi, 2 * np.pi) - np.pi


class PFCell(keras.layers.AbstractRNNCell):
    """
    PF-Net custom implementation for localization with RNN interface
    Implements the particle set update: observation, tramsition models and soft-resampling

    Cell inputs: observation, odometry
    Cell states: particle_states, particle_weights
    Cell outputs: particle_states, particle_weights (updated)
    """
    def __init__(self, params, is_igibson: bool, **kwargs):
        """
        :param params: parsed arguments
        """
        self.params = params

        self.states_shape = (self.params.batch_size, self.params.num_particles, 3)
        self.weights_shape = (self.params.batch_size, self.params.num_particles)
        self.map_shape = (self.params.batch_size, *self.params.global_map_size)
        super(PFCell, self).__init__(**kwargs)

        # self.particle_weights = None
        # self.particles = None

        self.is_igibson = is_igibson

        # models
        if self.params.likelihood_model == 'learned':
            self.obs_model = networks.obs_encoder(obs_shape=[56, 56, params.obs_ch])
            self.map_model = networks.map_encoder(map_shape=[28, 28, 1])
            self.joint_matrix_model = networks.map_obs_encoder()
            self.joint_vector_model = networks.likelihood_estimator()
        elif self.params.likelihood_model == 'scan_correlation':
            pass
        else:
            raise ValueError()

    # def reset_supervised(self, initial_particles, initial_particle_weights):
    #     self.particles = initial_particles
    #     self.particle_weights = initial_particle_weights

    @staticmethod
    def reset(robot_pose_pixel, env, params):
        # get random particles and weights based on init distribution conditions
        particles = tf.cast(tf.convert_to_tensor(
            get_random_particles(
                params.num_particles,
                params.init_particles_distr,
                tf.expand_dims(tf.convert_to_tensor(robot_pose_pixel, dtype=tf.float32), axis=0),
                env.trav_map,
                params.init_particles_cov,
                params.particles_range)), dtype=tf.float32)
        particle_weights = tf.constant(np.log(1.0 / float(params.num_particles)),
                                            shape=(params.batch_size, params.num_particles),
                                            dtype=tf.float32)
        return particles, particle_weights

    @staticmethod
    def compute_mse_loss(particles, particle_weights, true_states, trav_map_resolution):
        """
        Compute Mean Square Error (MSE) between ground truth pose and particles

        :param particle_states: particle states after observation update but before motion update (batch, trajlen, k, 3)
        :param particle_weights: particle likelihoods in the log space (unnormalized) (batch, trajlen, k)
        :param true_states: true state of robot (batch, trajlen, 3)
        :param float trav_map_resolution: The map rescale factor for iGibsonEnv

        :return dict: total loss and coordinate loss (in meters)
        """
        # assert particle_states.ndim == 4 and particle_weights.ndim == 3 and true_states.ndim == 3
        # TODO: what does this imply for scan-correlation likelihoods?
        # lin_weights = tf.nn.softmax(self.particle_weights, axis=-1)
        #
        # true_coords = true_states[..., :2]
        # mean_coords = tf.math.reduce_sum(tf.math.multiply(self.particles[..., :2], lin_weights[..., None]), axis=-2)


        est_pose = PFCell.get_est_pose(particles=particles, particle_weights=particle_weights)
        # coords_diffs = mean_coords - true_coords

        pose_diffs = est_pose - true_states

        # iGibsonEnv.scene.map_to_world()
        coords_diffs = pose_diffs[..., :2] * trav_map_resolution
        # coordinates loss component: (x-x')^2 + (y-y')^2
        loss_coords = tf.math.reduce_sum(tf.math.square(coords_diffs), axis=-1)

        # TODO: the paper describes this loss, but their code uses the thing below
        # orient_diffs = pose_diffs[..., 2]
        # loss_orient = tf.math.square(tf.math.reduce_sum(orient_diffs, axis=2))
        orient_diffs = particles[..., 2] - true_states[..., 2][..., None]
        # normalize between [-pi, +pi]
        orient_diffs = tf.math.floormod(orient_diffs + np.pi, 2 * np.pi) - np.pi
        # orientation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)
        loss_orient = tf.square(tf.reduce_sum(orient_diffs * lin_weights, axis=-1))

        loss_combined = loss_coords + 0.36 * loss_orient
        # loss_pred = tf.math.reduce_mean(loss_combined)

        loss = {}
        loss['pred'] = loss_combined  # [batch_size, trajlen]
        loss['coords'] = loss_coords  # [batch_size, trajlen]
        loss['orient'] = loss_orient  # [batch_size, trajlen]
        # loss['angular'] = tf.sqrt(tf.square(est_pose[..., 2] - norm_angle(true_states[..., 2])))

        return loss

    @staticmethod
    def bounding_box(img, robot_pose=None, lmt=100):
        """
        Bounding box of non-zeros in an array.

        :param img: numpy array
        :param robot_pose: numpy array of robot pose
        :param lmt: integer representing width/length of bounding box in pixels

        :return (int, int, int, int): bounding box indices top_row, bottom_row, left_column, right_column
        """
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        if robot_pose is not None:
            # futher constraint the bounding box
            x, y, _ = robot_pose

            rmin = np.rint(y - lmt) if (y - lmt) > rmin else rmin
            rmax = np.rint(y + lmt) if (y + lmt) < rmax else rmax
            cmin = np.rint(x - lmt) if (x - lmt) > cmin else cmin
            cmax = np.rint(x + lmt) if (x + lmt) < cmax else cmax

        return rmin, rmax, cmin, cmax,

    @staticmethod
    def get_est_pose(particles, particle_weights):
        """
        Compute estimate pose from particle and particle_weights (== weighted mean)
        """

        # batch_size = self.pf_params.batch_size
        # num_particles = self.pf_params.num_particles
        # particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)

        # assert list(particles.shape) == [batch_size, num_particles, 3], f'{particles.shape}'
        # assert list(lin_weights.shape) == [batch_size, num_particles], f'{lin_weights.shape}'

        est_pose_xy = tf.math.reduce_sum(tf.math.multiply(particles[..., :2], lin_weights[..., None]), axis=-2)
        # assert list(est_pose.shape) == [batch_size, 3], f'{est_pose.shape}'

        particle_theta_normed = norm_angle(particles[..., 2])
        est_pose_theta = tf.math.reduce_sum(tf.math.multiply(particle_theta_normed, lin_weights), axis=-1, keepdims=True)
        # normalize between [-pi, +pi]
        # part_x, part_y, part_th = tf.unstack(est_pose, axis=-1, num=3)  # (k, 3)
        # part_th = tf.math.floormod(part_th + np.pi, 2 * np.pi) - np.pi
        # est_pose = tf.stack([part_x, part_y, part_th], axis=-1)

        return tf.concat([est_pose_xy, est_pose_theta], axis=-1)

    @staticmethod
    def get_likelihood_map(particles, particle_weights, floor_map):
        """
        Construct Belief map/Likelihood map of particles & particle_weights from particle filter's current state

        :return likelihood_map_ext: [H, W, 4] map where each pixel position corresponds particle's position
            channel 0: floor_map of the environment
            channel 1: particle's weights
            channel 2, 3: particle's orientiation sine and cosine components
        """
        particles = particles[0].cpu().numpy()
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)[0].cpu().numpy()  # normalize weights
        likelihood_map_ext = np.zeros(list(floor_map.shape)[:2] + [4])  # [H, W, 4]

        # update obstacle map channel
        num_particles = particle_weights.shape[-1]
        likelihood_map_ext[:, :, 0] = np.squeeze(floor_map).copy() > 0  # clip to 0 or 1

        # x, y = (np.clip(int(particles[..., 0]), likelihood_map_ext.shape[0] - 1),
        #         np.clip(int(particles[..., 1]), likelihood_map_ext.shape[1] - 1))

        xy_clipped = np.clip(particles[..., :2], (0, 0), (likelihood_map_ext.shape[0] - 1, likelihood_map_ext.shape[1] - 1)).astype(int)

        for idx in range(num_particles):
            x, y = xy_clipped[idx]
            orn = particles[idx, 2]
            orn = datautils_norm_angle(orn)
            wt = lin_weights[idx]
            # x = np.clip(int(np.rint(col)), 0, likelihood_map_ext.shape[0] - 1)
            # y = np.clip(int(np.rint(row)), 0, likelihood_map_ext.shape[1] - 1)

            # update weights channel
            # likelihood_map_ext[
            #     int(np.rint(col-self.robot_size_px/2.)):int(np.rint(col+self.robot_size_px/2.))+1,
            #     int(np.rint(row-self.robot_size_px/2.)):int(np.rint(row+self.robot_size_px/2.))+1, 1] += wt
            likelihood_map_ext[x, y, 1] += wt

            # update orientation cos component channel
            # likelihood_map_ext[
            #     int(np.rint(col-self.robot_size_px/2.)):int(np.rint(col+self.robot_size_px/2.))+1,
            #     int(np.rint(row-self.robot_size_px/2.)):int(np.rint(row+self.robot_size_px/2.))+1, 2] += wt*np.cos(orn)
            likelihood_map_ext[x, y, 2] += wt * np.cos(orn)
            # update orientation sin component channel
            # likelihood_map_ext[
            #     int(np.rint(col-self.robot_size_px/2.)):int(np.rint(col+self.robot_size_px/2.))+1,
            #     int(np.rint(row-self.robot_size_px/2.)):int(np.rint(row+self.robot_size_px/2.))+1, 3] += wt*np.sin(orn)
            likelihood_map_ext[x, y, 3] += wt * np.sin(orn)
        # normalize: weighed mean of orientation channel w.r.t weights channel
        # indices = likelihood_map_ext[:, :, 1] > 0.
        # likelihood_map_ext[indices, 2] /= likelihood_map_ext[indices, 1]
        # likelihood_map_ext[indices, 3] /= likelihood_map_ext[indices, 1]

        return likelihood_map_ext


    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell
        :return tuple(TensorShapes): shape of particle_states, particle_weights
        """
        return [tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:]), tf.TensorShape(self.map_shape[1:])]

    @property
    def output_size(self):
        """
        Size(s) of output(s) produced by this cell
        :return tuple(TensorShapes): shape of particle_states, particle_weights
        """
        return [tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:]), tf.TensorShape(self.map_shape[1:])]

    def call(self, input, state):
        """
        Implements a particle update
        :param input: observation (batch, 56, 56, ch), odometry (batch, 3), global_map (batch, H, W, 1)
            observation is the sensor reading at time t,
            odometry is the relative motion from time t to t+1,
            global map of environment
        :param state: particle_states (batch, k, 3), particle_weights (batch, k)
            weights are assumed to be in log space and unnormalized
        :return output: particle_states and particle_weights after the observation update.
            (but before the transition update)
        :return state: updated particle_states and particle_weights.
            (but after both observation and transition updates)
        """
        particles, particle_weights, global_map = state
        observation, odometry = input

        if self.is_igibson:
            # motion update: odometry is motion from t-1 to t
            particles = self.transition_model(particles, odometry)

        # observation update
        # particle_weights, particles = self.particle_weights, self.particles

        if self.params.likelihood_model == 'learned':
            lik = self.observation_update(global_map, particles, observation)
        else:
            lik = self.observation_update_scan_correlation(global_map=global_map, particle_states=particles, observation=observation)
        particle_weights = particle_weights + lik  # unnormalized

        # resample
        if self.params.resample:
            particles, particle_weights = self.resample(particles, particle_weights, alpha=self.params.alpha_resample_ratio)

        # construct output before motion update
        output = [particles, particle_weights]

        # motion update which affect the particle state input at next step
        if not self.is_igibson:
            particles = self.transition_model(particles, odometry)

        # construct new state after motion update
        # state = [particle_states, particle_weights, global_map]
        # print('output shapes:', [output[0].shape, output[1].shape])
        # print('state shapes:', [state[0].shape, state[1].shape, state[2].shape])
        # print('output types:', [output[0].dtype, output[1].dtype])
        # print('state types:', [state[0].dtype, state[1].dtype, state[2].dtype])
        #
        # print("output nans:", [tf.print(tf.reduce_any(tf.math.is_nan(o))) for o in output])
        # print("state nans:", [tf.print(tf.reduce_any(tf.math.is_nan(o))) for o in state])

        # self.particle_weights = particle_weights
        # self.particles = particles

        state = [particles, particle_weights, global_map]

        return output, state

    def observation_update_scan_correlation(self, global_map, particle_states, observation):
        """
        :param observation: image observation (batch, 56, 56, ch)
        """
        return self.calc_scan_correlation(global_map=global_map, particle_states=particle_states, observation=observation, window_scaler=self.params.window_scaler, scan_size=128, is_igibson=self.is_igibson)

    @staticmethod
    def calc_scan_correlation(global_map, particle_states, observation, window_scaler, scan_size: float, is_igibson: bool):
        # NOTE: occupancy_grid covers 5mx5m with 128x128pixels -> resolution 5/128 = 0.0390625
        # global map is at resolution 0.1
        # -> scale global map by 0.1/0.0390625 = 0.390625?
        window_scaler = 0.390625
        # TODO: this also needs to be done in the original pfnet update / respectively anywhere that calls PFCell.transform_maps()
        # global_map = np.flip(global_map, axis=-3)

        batch_size, num_particles = particle_states.shape.as_list()[:2]

        # transform global maps to local maps
        # in [0, 2] range?
        local_maps = PFCell.transform_maps(global_map, particle_states, (scan_size, scan_size), window_scaler, agent_at_bottom=False, flip_map=is_igibson)

        # flatten batch and particle dimensions
        local_maps = tf.reshape(local_maps, [batch_size * num_particles, -1])

        # observation: 0: occupied, 0.5: unexplored, 1.0: free
        # -> reformat to 0: free, 1: occupied, 2: unexplored
        grid = tf.zeros(observation.shape, dtype=tf.int32)
        o = tf.cast(2 * observation, tf.int32)
        grid += tf.where(o == 1, 2, 0)
        grid += tf.where(o == 0, 1, 0)

        # NOTE: orientation on occupancy grid is originally to the right. Change to be looking upwards as in the local_maps
        grid = tf.image.rot90(grid, 1)




        def _compare_grids(gt_pose=(55.   , 47.   ,  3.068), rotate_grid=0, flip=False):
            ps = tf.concat([tf.convert_to_tensor(list(gt_pose), dtype=tf.float32)[None, None],
                            particle_states[:, 1:]], 1)
            local_maps = PFCell.transform_maps(global_map, ps, (scan_size, scan_size), window_scaler, agent_at_bottom=False, flip_map=self.is_igibson)
            # flatten batch and particle dimensions
            local_maps = tf.reshape(local_maps, [batch_size * num_particles, -1])

            from matplotlib import pyplot as plt
            import cv2
            f, ax = plt.subplots(1, 3)
            g = grid[0]
            g = np.rot90(g, rotate_grid)
            l = np.reshape(local_maps[0], (128, 128))
            if flip:
                l = np.fliplr(l)
            ax[0].imshow(cv2.cvtColor((g * 255 / 2).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            ax[1].imshow(l)
            for i, c in enumerate(['g', 'w', 'w']):
                ax[i].set_xticks(np.arange(0, 128, 32))
                ax[i].set_yticks(np.arange(0, 128, 32))
                ax[i].grid(color=c, linestyle='-', linewidth=1)

            ax[2].imshow(np.squeeze(np.array(g == 1, dtype=int)) - np.array(l > 0, dtype=int))
            plt.show()

        def _plot_global_map():
            from matplotlib import pyplot as plt
            plt.imshow(global_map[0]);
            ax = plt.gca();
            ax.set_xticks(np.arange(0, 100, 10))
            ax.set_yticks(np.arange(0, 100, 10))
            plt.grid(color='w', linestyle='-', linewidth=1), plt.show()

        # _compare_grids((59., 60., -0.304), flip=False)

        # with tf.print(tf.unique(tf.reshape(grid, -1))):
        grid_tiled = tf.tile(tf.expand_dims(grid, axis=1), [1, num_particles, 1, 1, 1])
        grid_tiled = tf.reshape(grid_tiled, [batch_size * num_particles, -1])

        # NOTE: will only work if not downsampled!
        # mask out the unexplored parts of the observation
        tf.assert_equal(batch_size, 1,
                        "boolean_mask won't work easily with batches as it should mask a different number of elements per batch")
        explored = (grid != 2)
        mask = tf.reshape(explored, [scan_size ** 2])
        local_maps_explored = tf.boolean_mask(local_maps, mask, axis=1)
        grid_tiled_explored = tf.boolean_mask(grid_tiled, mask, axis=1)

        correlation = tfp.stats.correlation(local_maps_explored, tf.cast(grid_tiled_explored, tf.float32),
                                            sample_axis=-1, event_axis=None)
        # if std of either masked part is 0, we get nan values -> set to lowest value, i.e. -1
        correlation = tf.where(tf.math.is_nan(correlation), -1, correlation)
        
        # correlation = cosine_similarity(local_maps_explored, tf.cast(grid_tiled_explored, tf.float32))

        lik = tf.reshape(correlation, [batch_size, num_particles])
        # ensure all positive
        lik -= tf.minimum(0.0, tf.reduce_min(lik, -1))
        lik = lik / tf.reduce_sum(lik, axis=-1)

        # TODO: should be unnormalized / in log space
        log_lik = tf.math.log(lik)
        log_lik = tf.where(tf.math.is_inf(log_lik), -7, log_lik)

        return log_lik

    @tf.function(jit_compile=True)
    def observation_update(self, global_map, particle_states, observation):
        """
        Implements a discriminative observation model for localization
        The model transforms global map to local maps for each particle,
        where a local map is a local view from state defined by the particle.
        :param global_map: global map input (batch, None, None, ch)
            assumes range[0, 2] were 0: occupied and 2: free space
        :param particle_states: particle states before observation update (batch, k, 3)
        :param observation: image observation (batch, 56, 56, ch)
        :return (batch, k): particle likelihoods in the log space (unnormalized)
        """
        if self.params.obs_mode == "occupancy_grid":
            # robot is looking to the right, but should be looking up
            observation = tf.image.rot90(observation, 1)

        batch_size, num_particles = particle_states.shape.as_list()[:2]

        # transform global maps to local maps
        # TODO: only set agent_at_bottom true if using [rgb, d], not for lidar?
        local_maps = PFCell.transform_maps(global_map, particle_states, (28, 28), self.params.window_scaler, agent_at_bottom=True, flip_map=self.is_igibson)

        # rescale from [0, 2] to [-1, 1]    -> optional
        local_maps = -(local_maps - 1)

        # flatten batch and particle dimensions
        local_maps = tf.reshape(local_maps, [batch_size * num_particles] + local_maps.shape.as_list()[2:])

        # get features from local maps
        map_features = self.map_model(local_maps)

        # get features from observation
        obs_features = self.obs_model(observation)

        # tile observation features
        obs_features = tf.tile(tf.expand_dims(obs_features, axis=1), [1, num_particles, 1, 1, 1])
        obs_features = tf.reshape(obs_features, [batch_size * num_particles] + obs_features.shape.as_list()[2:])

        # sanity check
        assert obs_features.shape.as_list()[:-1] == map_features.shape.as_list()[:-1]

        # merge map and observation features
        joint_features = tf.concat([map_features, obs_features], axis=-1)
        joint_features = self.joint_matrix_model(joint_features)

        # reshape to a vector
        joint_features = tf.reshape(joint_features, [batch_size * num_particles, -1])
        lik = self.joint_vector_model(joint_features)
        lik = tf.reshape(lik, [batch_size, num_particles])

        return lik

    @tf.function(jit_compile=True)
    def resample(self, particle_states, particle_weights, alpha):
        """
        Implements soft-resampling of particles
        :param particle_states: particle states (batch, k, 3)
        :param particle_weights: unnormalized particle weights in log space (batch, k)
        :param alpha: trade-off parameter for soft-resampling
            alpha == 1, corresponds to standard hard-resampling
            alpha == 0, corresponds to sampling particles uniformly ignoring weights
        :return (batch, k, 3) (batch, k): resampled particle states and particle weights
        """

        assert 0.0 < alpha <= 1.0
        batch_size, num_particles = particle_states.shape.as_list()[:2]

        # normalize weights
        particle_weights = particle_weights - tf.math.reduce_logsumexp(particle_weights, axis=-1, keepdims=True)

        # sample uniform weights
        uniform_weights = tf.constant(np.log(1.0/float(num_particles)), shape=(batch_size, num_particles), dtype=tf.float32)

        # build sample distribution q(s) and update particle weights
        if alpha < 1.0:
            # soft-resampling
            q_weights = tf.stack([particle_weights + np.log(alpha),
                                  uniform_weights + np.log(1.0 - alpha)],
                                  axis=-1)
            q_weights = tf.math.reduce_logsumexp(q_weights, axis=-1, keepdims=False)
            q_weights = q_weights - tf.reduce_logsumexp(q_weights, axis=-1, keepdims=True) # normalized

            particle_weights = particle_weights - q_weights  # unnormalized
        else:
            # hard-resampling -> produces zero gradients
            q_weights = particle_weights
            particle_weights = uniform_weights

        # sample particle indices according to q(s)
        indices = tf.random.categorical(q_weights, num_particles, dtype=tf.int32)  # shape: (bs, k)

        # index into particles
        helper = tf.range(0, batch_size*num_particles, delta=num_particles, dtype=tf.int32)  # (batch, )
        indices = indices + tf.expand_dims(helper, axis=1)

        particle_states = tf.reshape(particle_states, (batch_size * num_particles, 3))
        particle_states = tf.gather(particle_states, indices=indices, axis=0)  # (bs, k, 3)

        particle_weights = tf.reshape(particle_weights, (batch_size * num_particles, ))
        particle_weights = tf.gather(particle_weights, indices=indices, axis=0)  # (bs, k)

        return particle_states, particle_weights

    @tf.function(jit_compile=True)
    def transition_model(self, particle_states, odometry):
        """
        Implements a stochastic transition model for localization
        :param particle_states: particle states before motion update (batch, k, 3)
        :param odometry: odometry reading - relative motion in robot coordinate frame (batch, 3)
        :return (batch, k, 3): particle states updated with the odometry and optionally transition noise
        """

        translation_std = self.params.transition_std[0]   # in pixels
        rotation_std = self.params.transition_std[1]    # in radians

        part_x, part_y, part_th = tf.unstack(particle_states, axis=-1, num=3)   # (bs, k, 3)

        # non-noisy odometry
        odometry = tf.expand_dims(odometry, axis=1) # (batch_size, 1, 3)
        odom_x, odom_y, odom_th = tf.unstack(odometry, axis=-1, num=3)

        # sample noisy orientation
        noise_th = tf.random.normal(part_th.get_shape(), mean=0.0, stddev=1.0) * rotation_std

        # add orientation noise before translation
        part_th = part_th + noise_th

        # non-noisy translation and rotation
        cos_th = tf.cos(part_th)
        sin_th = tf.sin(part_th)
        delta_x = cos_th * odom_x - sin_th * odom_y
        delta_y = sin_th * odom_x + cos_th * odom_y
        delta_th = odom_th

        # sample noisy translation
        delta_x = delta_x + tf.random.normal(delta_x.get_shape(), mean=0.0, stddev=1.0) * translation_std
        delta_y = delta_y + tf.random.normal(delta_y.get_shape(), mean=0.0, stddev=1.0) * translation_std

        return tf.stack([part_x + delta_x, part_y + delta_y, part_th + delta_th], axis=-1)   # (bs, k, 3)

    @staticmethod
    @tf.function(jit_compile=True)
    def transform_maps(global_map, particle_states, local_map_size, window_scaler=None, agent_at_bottom: bool = True, flip_map: bool = False):
        """
        Implements global to local map transformation
        :param global_map: global map input (batch, None, None, ch)
        :param particle_states: particle states that define local view for transformation (batch, k, 3)
        :param local_map_size: size of output local maps (height, width)
        :param window_scaler: global map will be down-scaled by some int factor
        :return (batch, k, local_map_size[0], local_map_size[1], ch): each local map shows different transformation
            of global map corresponding to particle state
        """

        # flatten batch and particle
        batch_size, num_particles = particle_states.shape.as_list()[:2]
        total_samples = batch_size * num_particles
        flat_states = tf.reshape(particle_states, [total_samples, 3])

        # NOTE: For igibson, first value indexes the y axis, second the x axis
        if flip_map:
            flat_states = tf.gather(flat_states, [1, 0, 2], axis=-1)

        # define variables
        # TODO: could resize the map before doing the affine transform, instead of doing it at the same time

        input_shape = tf.shape(global_map)
        global_height = tf.cast(input_shape[1], tf.float32)
        global_width = tf.cast(input_shape[2], tf.float32)
        height_inverse = 1.0 / global_height
        width_inverse = 1.0 / global_width
        zero = tf.constant(0, dtype=tf.float32, shape=(total_samples, ))
        one = tf.constant(1, dtype=tf.float32, shape=(total_samples, ))

        # normalize orientations and precompute cos and sin functions
        theta = -flat_states[:, 2] - 0.5 * np.pi
        costheta = tf.cos(theta)
        sintheta = tf.sin(theta)

        # construct affine transformation matrix step-by-step

        # 1: translate the global map s.t. the center is at the particle state
        translate_x = (flat_states[:, 0] * width_inverse * 2.0) - 1.0
        translate_y = (flat_states[:, 1] * height_inverse * 2.0) - 1.0

        transm1 = tf.stack((one, zero, translate_x, zero, one, translate_y, zero, zero, one), axis=1)
        transm1 = tf.reshape(transm1, [total_samples, 3, 3])

        # 2: rotate map s.t the orientation matches that of the particles
        rotm = tf.stack((costheta, sintheta, zero, -sintheta, costheta, zero, zero, zero, one), axis=1)
        rotm = tf.reshape(rotm, [total_samples, 3, 3])

        # 3: scale down the map
        if window_scaler is not None:
            scale_x = tf.fill((total_samples, ), float(local_map_size[1] * window_scaler) * width_inverse)
            scale_y = tf.fill((total_samples, ), float(local_map_size[0] * window_scaler) * height_inverse)
        else:
            # identity
            scale_x = one
            scale_y = one

        scalem = tf.stack((scale_x, zero, zero, zero, scale_y, zero, zero, zero, one), axis=1)
        scalem = tf.reshape(scalem, [total_samples, 3, 3])

        # finally chain all traformation matrices into single one
        transform_m = tf.matmul(tf.matmul(transm1, rotm), scalem)
        # 4: translate the local map s.t. the particle defines the bottom mid_point instead of the center
        if agent_at_bottom:
            translate_y2 = tf.constant(-1.0, dtype=tf.float32, shape=(total_samples, ))

            transm2 = tf.stack((one, zero, zero, zero, one, translate_y2, zero, zero, one), axis=1)
            transm2 = tf.reshape(transm2, [total_samples, 3, 3])

            transform_m = tf.matmul(transform_m, transm2)

        # reshape to format expected by spatial transform network
        transform_m = tf.reshape(transform_m[:, :2], [batch_size, num_particles, 6])

        # iterate over num_particles to transform image using spatial transform network
        def transform_batch(U, thetas, out_size):
            num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
            indices = [[i] * num_transforms for i in range(num_batch)]
            input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
            return transformer(input_repeated, thetas, out_size)

        # t = time.time()
        if batch_size == 1 and global_map.shape[2] <= 100:
            # assert batch_size == 1, (batch_size, "Just haven't tested it for batches yet, might already work though")
            # with tf.device('CPU'):
            local_maps_new = transform_batch(U=global_map, thetas=transform_m, out_size=local_map_size)
            local_maps = local_maps_new[tf.newaxis]
        else:
            # create mini batches to not run out of vram
            # min_batch_size = 1
            # lmaps = []
            # for b in np.arange(0, batch_size, min_batch_size):
            #     # print(b, b+min_batch_size, global_map[b:b+min_batch_size].shape)
            #     lmaps.append(transform_batch(U=global_map[b:b+min_batch_size], thetas=transform_m[b:b+min_batch_size], out_size=local_map_size))
            # local_maps = tf.concat(lmaps, 0)
            # print(f"ccccccccccc {(time.time() - t) / 60:.3f}")

            # t = time.time()
            local_maps = tf.stack([
                transformer(global_map, transform_m[:, i], local_map_size) for i in range(num_particles)
            ], axis=1)
            # print(f"zzzzzzzzzzzzzzz {(time.time() - t) / 60:.3f}")



        # reshape if any information has lost in spatial transform network
        local_maps = tf.reshape(local_maps, [batch_size, num_particles, local_map_size[0], local_map_size[1], global_map.shape.as_list()[-1]])


        # NOTE: flip to have the same alignment as the other modalities
        if flip_map:
            local_maps = tf.experimental.numpy.flip(local_maps, -2)

        # particle_states = tf.convert_to_tensor([[65.   , 59.   ,  1.538]])[None]
        # def _plot():
        #     import matplotlib.pyplot as plt
        #     b = 0
        #     f, ax = plt.subplots(1, 1)
        #     fmap, ax_map = plt.subplots(1, 1)
        #     ax_map.imshow(global_map[b])
        #     for i, a in enumerate(np.reshape(ax, -1)):
        #         ax_map.scatter(particle_states[b, i, 1], particle_states[b, i, 0], marker='o', s=1)
        #         a.imshow(local_maps[b, i])
        #     plt.show()


        return local_maps   # (batch_size, num_particles, 28, 28, 1)


def pfnet_model(params, is_igibson: bool):

    # batch_size = params.batch_size
    # num_particles = params.num_particles
    # global_map_size = params.global_map_size
    # trajlen = params.trajlen
    if hasattr(params, 'obs_ch'):
        obs_ch = params.obs_ch
    else:
        obs_ch = params.obs_ch = 3

    if params.likelihood_model == "scan_correlation":
        sz = 128
    else:
        sz = 56
        
    observation = keras.Input(shape=[params.trajlen, sz, sz, obs_ch], batch_size=params.batch_size)   # (bs, T, 56, 56, C)
    odometry = keras.Input(shape=[params.trajlen, 3], batch_size=params.batch_size)    # (bs, T, 3)

    global_map = keras.Input(shape=params.global_map_size, batch_size=params.batch_size)   # (bs, H, W, 1)
    particle_states = keras.Input(shape=[params.num_particles, 3], batch_size=params.batch_size)   # (bs, k, 3)
    particle_weights = keras.Input(shape=[params.num_particles], batch_size=params.batch_size)    # (bs, k)

    cell = PFCell(params, is_igibson=is_igibson)
    rnn = keras.layers.RNN(cell, return_sequences=True, return_state=params.return_state, stateful=False)

    state = [particle_states, particle_weights, global_map]
    input = (observation, odometry)
    
    # x = rnn(inputs=tuple([tf.random.uniform(i.shape) for i in input]), 
    #         initial_state=tuple([tf.random.uniform(i.shape) for i in state]))
    x = rnn(inputs=input, initial_state=state)
    output, out_state = x[:2], x[2:]
    # output = [keras.Input(o.shape[1:], batch_size=params.batch_size) for o in output]
    # out_state = [keras.Input(o.shape[1:], batch_size=params.batch_size) for o in out_state]

    return keras.Model(
        inputs=([observation, odometry], state),
        outputs=([output, out_state])
    )
    # return cell

if __name__ == '__main__':
    # obs_model = observation_model()
    # keras.utils.plot_model(obs_model, to_file='obs_model.png', show_shapes=True, dpi=64)
    #
    # observations = np.random.random((8*10, 56, 56, 3))
    # obs_out = obs_model(observations)
    # print(obs_out.shape)
    #
    # map_model = map_model()
    # keras.utils.plot_model(map_model, to_file='map_model.png', show_shapes=True, dpi=64)
    #
    # local_maps = np.random.random((8*10, 28, 28, 1))
    # map_out = map_model(local_maps)
    # print(map_out.shape)
    #
    # joint_matrix_model = joint_matrix_model()
    # keras.utils.plot_model(joint_matrix_model, to_file='joint_matrix_model.png', show_shapes=True, dpi=64)
    #
    # joint_features = tf.concat([map_out, obs_out], axis=-1)
    # joint_matrix_out = joint_matrix_model(joint_features)
    # print(joint_matrix_out.shape)
    #
    # joint_vector_model = joint_vector_model()
    # keras.utils.plot_model(joint_vector_model, to_file='joint_vector_model.png', show_shapes=True, dpi=64)
    #
    # joint_matrix_out = tf.reshape(joint_matrix_out, (8 * 10, -1))
    # joint_vector_out = joint_vector_model(joint_matrix_out)
    # joint_vector_out = tf.reshape(joint_vector_out, [8, 10])
    # print(joint_vector_out.shape)
    #
    # particle_states = tf.random.uniform((8, 10, 3))
    # odometry = tf.random.uniform((8, 3))
    # transition_std = np.array([0.0, 0.0])
    # map_pixel_in_meters = 0.02
    # transition_out = transition_model(particle_states, odometry, (8, 10), transition_std, map_pixel_in_meters)
    # print(transition_out.shape)
    #
    # global_maps = tf.random.uniform((8, 300, 300, 1))
    # transform_out = transform_maps(global_maps, particle_states, (8, 10), (28, 28))
    # print(transform_out.shape)

    argparser = argparse.ArgumentParser()
    params = argparser.parse_args()

    params.transition_std = np.array([0.0, 0.0])
    params.map_pixel_in_meters = 0.02
    params.batch_size = 8
    params.num_particles = 30
    params.time_steps = 5

    model = pfnet_model(params)
    keras.utils.plot_model(model, to_file='pfnet_model.png', show_shapes=True, dpi=64)

    particle_states = tf.random.uniform((params.batch_size, params.num_particles, 3))
    particle_weights = tf.random.uniform((params.batch_size, params.num_particles))
    observation = tf.random.uniform((params.batch_size, params.time_steps, 56, 56, 3))
    odometry = tf.random.uniform((params.batch_size, params.time_steps, 3))
    global_map = tf.random.uniform((params.batch_size, params.time_steps, 100, 100, 1))
    inputs = ([observation, odometry, global_map], [particle_states, particle_weights])
    output, state = model(inputs)
    print(output[0].shape, output[1].shape, state[0].shape, state[1].shape)

    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint')

    # Create a new model instance
    new_model = pfnet_model(params)

    # Restore the weights
    new_model.load_weights('./checkpoints/my_checkpoint')
