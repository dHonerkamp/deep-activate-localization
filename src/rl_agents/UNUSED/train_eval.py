#!/usr/bin/env python3

"""
reference:
    https://github.com/StanfordVL/agents/blob/cvpr21_challenge_tf2.4/tf_agents/agents/sac/examples/v2/train_eval.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import numpy as np
import random

# tf.config.run_functions_eagerly(False)

import functools

# custom tf_agents
import rl_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_step_driver
# from tf_agents.environments import suite_gibson
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import py_metrics
from tf_agents.metrics import batched_py_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from pfnetwork.arguments import parse_common_args
from pfnetwork.train import WANDB_PROJECT, init_pfnet_model
import wandb
from pathlib import Path
from environments import suite_gibson

# flags.DEFINE_string('obs_mode', 'rgb-depth', 'Observation input type. Possible values: rgb / depth / rgb-depth / occupancy_grid.')
# flags.DEFINE_list('custom_output', ['rgb_obs', 'depth_obs', 'occupancy_grid', 'floor_map', 'kmeans_cluster', 'likelihood_map'], 'A comma-separated list of env observation types.')
# flags.DEFINE_integer('obs_ch', 4, 'Observation channels.')
# flags.DEFINE_integer('num_clusters', 10, 'Number of particle clusters.')
flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'Root directory for writing logs/summaries/checkpoints.')
# flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
# flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

# flags.DEFINE_integer('num_iterations', 1000000, 'Total number train/eval iterations to perform.')
# flags.DEFINE_integer('initial_collect_steps', 1000, 'Number of steps to collect at the beginning of training using random policy')
# flags.DEFINE_integer('collect_steps_per_iteration', 1, 'Number of steps to collect and be added to the replay buffer after every training iteration')
# flags.DEFINE_integer('num_parallel_environments', 1, 'Number of environments to run in parallel')
# flags.DEFINE_integer('num_parallel_environments_eval', 1, 'Number of environments to run in parallel for eval')
# flags.DEFINE_integer('replay_buffer_capacity', 1000000, 'Replay buffer capacity per env.')
# flags.DEFINE_integer('train_steps_per_iteration', 1, 'Number of training steps in every training iteration')
# flags.DEFINE_integer('batch_size', 256, 'Batch size for each training step. For each training iteration, we first collect collect_steps_per_iteration steps to the replay buffer. Then we sample batch_size steps from the replay buffer and train the model for train_steps_per_iteration times.')
# flags.DEFINE_float('gamma', 0.99, 'Discount_factor for the environment')
# flags.DEFINE_float('actor_learning_rate', 3e-4, 'Actor learning rate')
# flags.DEFINE_float('critic_learning_rate', 3e-4, 'Critic learning rate')
# flags.DEFINE_float('alpha_learning_rate', 3e-4, 'Alpha learning rate')
# flags.DEFINE_integer('seed', 100, 'random seed')

# flags.DEFINE_boolean('use_tf_functions', True, 'Whether to use graph/eager mode execution')
# flags.DEFINE_boolean('use_parallel_envs', False, 'Whether to use parallel env or not')
# flags.DEFINE_integer('num_eval_episodes', 10, 'The number of episodes to run eval on.')
# flags.DEFINE_integer('eval_interval', 10000, 'Run eval every eval_interval train steps')
# flags.DEFINE_boolean('eval_only', False, 'Whether to run evaluation only on trained checkpoints')
# flags.DEFINE_boolean('eval_deterministic', False, 'Whether to run evaluation using a deterministic policy')
# flags.DEFINE_integer('gpu_c', 0, 'GPU id for compute, e.g. Tensorflow.')

# Added for Gibson
# flags.DEFINE_boolean('is_localize_env', True, 'Whether to use navigation/localization env')
# flags.DEFINE_string('config_file', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'turtlebot_point_nav.yaml'), 'Config file for the experiment.')
# flags.DEFINE_list('model_ids', None, 'A comma-separated list of model ids to overwrite config_file. len(model_ids) == num_parallel_environments')
# flags.DEFINE_list('model_ids_eval', None, 'A comma-separated list of model ids to overwrite config_file for eval. len(model_ids) == num_parallel_environments_eval')
# flags.DEFINE_string('env_mode', 'headless', 'Mode for the simulator (gui or headless)')
# flags.DEFINE_float('action_timestep', 1.0 / 10.0, 'Action timestep for the simulator')
# flags.DEFINE_float('physics_timestep', 1.0 / 40.0, 'Physics timestep for the simulator')
# flags.DEFINE_integer('gpu_g', 0, 'GPU id for graphics, e.g. Gibson.')

# Added for Particle Filter
# flags.DEFINE_boolean('init_env_pfnet', False, 'Whether to initialize particle filter net')
# flags.DEFINE_string('init_particles_distr', 'gaussian', 'Distribution of initial particles. Possible values: gaussian / uniform.')
# flags.DEFINE_list('init_particles_std', [0.15, 0.523599], 'Standard deviations for generated initial particles for tracking distribution. Values: translation std (meters), rotation std (radians)')
# flags.DEFINE_integer('particles_range', 100, 'Pixel range to limit uniform distribution sampling +/- box particles_range center at robot pose')
flags.DEFINE_integer('num_particles', 500, 'Number of particles in Particle Filter.')
flags.DEFINE_boolean('resample', True, 'Resample particles in Particle Filter. Possible values: true / false.')
flags.DEFINE_float('alpha_resample_ratio', 0.5, 'Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true. Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling.')
flags.DEFINE_list('transition_std', [0.02, 0.0872665], 'Standard deviations for transition model. Values: translation std (meters), rotation std (radians)')
# flags.DEFINE_list('global_map_size', [100, 100, 1], 'Global map size in pixels (H, W, C).')
# flags.DEFINE_float('window_scaler', 1.0, 'Rescale factor for extracing local map.')
# flags.DEFINE_string('pfnet_load', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pfnetwork/checkpoints/pfnet_igibson_data', 'checkpoint_87_5.830/pfnet_checkpoint'), 'Load a previously trained pfnet model from a checkpoint file.')
# flags.DEFINE_boolean('use_plot', False, 'Enable Plotting of particles on env map. Possible values: true / false.')
# flags.DEFINE_boolean('store_plot', False, 'Store the plots sequence as video. Possible values: true / false.')
FLAGS = flags.FLAGS


def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


# @gin.configurable
def train_eval(
        root_dir,
        gpu=0,
        env_load_fn=None,
        model_ids=None,
        reload_interval=None,
        eval_env_mode='headless',
        num_iterations=1000000,
        conv_1d_layer_params=None,
        conv_2d_layer_params=None,
        encoder_fc_layers=[256],
        actor_fc_layers=[256, 256],
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=[256, 256],
        # Params for collect
        initial_collect_steps=10000,
        collect_steps_per_iteration=1,
        num_parallel_environments=1,
        replay_buffer_capacity=1000000,
        # Params for target update
        target_update_tau=0.005,
        target_update_period=1,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        use_tf_functions=True,
        use_parallel_envs=True,
        # Params for eval
        num_eval_episodes=30,
        eval_interval=10000,
        eval_only=False,
        eval_deterministic=False,
        num_parallel_environments_eval=1,
        model_ids_eval=None,
        # Params for summaries and logging
        train_checkpoint_interval=10000,
        policy_checkpoint_interval=10000,
        rb_checkpoint_interval=50000,
        log_interval=100,
        summary_interval=1000,
        summaries_flush_secs=10,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):
    """A simple train and eval for SAC."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    # tf.profiler.experimental.start(logdir='./log_dir')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
                    tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)]
    eval_info_metrics = [rl_utils.AverageStepPositionErrorMetric(buffer_size=num_eval_episodes),
                         rl_utils.AverageStepOrientationErrorMetric(buffer_size=num_eval_episodes),
                         rl_utils.AverageStepCollisionPenalityMetric(buffer_size=num_eval_episodes)]

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        if model_ids is None:
            model_ids = [None] * num_parallel_environments
        else:
            assert len(model_ids) == num_parallel_environments, 'model ids provided, but length not equal to num_parallel_environments'

        if model_ids_eval is None:
            model_ids_eval = [None] * num_parallel_environments_eval
        else:
            assert len(model_ids_eval) == num_parallel_environments_eval, 'model ids eval provided, but length not equal to num_parallel_environments_eval'

        if use_parallel_envs:
            tf_py_env = [lambda model_id=model_ids[i]: env_load_fn(model_id, 'headless', gpu) for i in range(num_parallel_environments)]
            tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(tf_py_env))

            if eval_env_mode == 'gui':
                assert num_parallel_environments_eval == 1, 'only one GUI env is allowed'
            eval_py_env = [lambda model_id=model_ids_eval[i]: env_load_fn(model_id, eval_env_mode, gpu) for i in range(num_parallel_environments_eval)]
            eval_tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(eval_py_env))
        else:
            # HACK: use same env for train and eval
            tf_py_env = env_load_fn(model_ids[0], 'headless', True, gpu)
            tf_env = tf_py_environment.TFPyEnvironment(tf_py_env)
            eval_tf_env = tf_env

        time_step_spec = tf_env.time_step_spec()
        observation_spec = time_step_spec.observation
        action_spec = tf_env.action_spec()
        logging.info('\n Observation specs: %s \n Action specs: %s', observation_spec, action_spec)

        glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        preprocessing_layers = {}
        if 'rgb_obs' in observation_spec:
            preprocessing_layers['rgb_obs'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'depth_obs' in observation_spec:
            preprocessing_layers['depth_obs'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'floor_map' in observation_spec:
            preprocessing_layers['floor_map'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'likelihood_map' in observation_spec:
            preprocessing_layers['likelihood_map'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'scan' in observation_spec:
            preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=conv_1d_layer_params,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'task_obs' in observation_spec:
            preprocessing_layers['task_obs'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'kmeans_cluster' in observation_spec:
            preprocessing_layers['kmeans_cluster'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if len(preprocessing_layers) <= 1:
            preprocessing_combiner = None
        else:
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=actor_fc_layers,
            # continuous_projection_net=normal_projection_net,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,
            kernel_initializer=glorot_uniform_initializer
        )
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            kernel_initializer=glorot_uniform_initializer
        )

        logging.info('Creating SAC Agent')
        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)
        tf_agent.initialize()

        # Make the replay buffer.
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=tf_agent.collect_data_spec,
                                                                       batch_size=tf_env.batch_size,
                                                                       max_length=replay_buffer_capacity)
        replay_observer = [replay_buffer.add_batch]

        if eval_deterministic:
            eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
        else:
            eval_policy = tf_agent.policy

        train_metrics = [tf_metrics.NumberOfEpisodes(),
                         tf_metrics.EnvironmentSteps(),
                         tf_metrics.AverageReturnMetric(buffer_size=100, batch_size=tf_env.batch_size),
                         tf_metrics.AverageEpisodeLengthMetric(buffer_size=100, batch_size=tf_env.batch_size),]

        initial_collect_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
        collect_policy = tf_agent.collect_policy

        train_checkpointer = common.Checkpointer(ckpt_dir=train_dir,
                                                 agent=tf_agent,
                                                 global_step=global_step,
                                                 metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(train_dir, 'policy'),
                                                  policy=eval_policy,
                                                  global_step=global_step)
        rb_checkpointer = common.Checkpointer(ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
                                              max_to_keep=1,
                                              replay_buffer=replay_buffer)

        train_checkpointer.initialize_or_restore()
        rb_checkpointer.initialize_or_restore()

        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=initial_collect_steps)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=collect_steps_per_iteration)

        if use_tf_functions:
            # initial_collect_driver.run = common.function(initial_collect_driver.run)
            # collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        if replay_buffer.num_frames() == 0:
            # Collect initial replay data.
            logging.info('Initializing replay buffer by collecting experience for %d steps with a random policy.', initial_collect_steps)
            initial_collect_driver.run()

        results = metric_utils.eager_compute(
            eval_metrics,
            eval_info_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
            # use_function=use_tf_functions,
            use_function=False,
        )
        if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics+eval_info_metrics)

        if eval_only:
            logging.info('Eval finished')
            return
        logging.info('Training starting .....')

        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        # Prepare replay buffer as dataset with invalid transitions filtered.
        def _filter_invalid_transition(trajectories, unused_arg1):
            return ~trajectories.is_boundary()[0]

        dataset = replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2).unbatch().filter( _filter_invalid_transition).batch(batch_size).prefetch(5)
        # Dataset generates trajectories with shape [Bx2x...]
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator)
            # if not use_tf_functions:
            #     tf.debugging.check_numerics(experience, "Bad!")
            return tf_agent.train(experience)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        if use_tf_functions:
            train_step = common.function(train_step)

        global_step_val = global_step.numpy()
        while global_step_val < num_iterations:
            start_time = time.time()
            time_step, policy_state = collect_driver.run(time_step=time_step, policy_state=policy_state)
            for _ in range(train_steps_per_iteration):
                train_loss = train_step()
            time_acc += time.time() - start_time

            global_step_val = global_step.numpy()

            if global_step_val % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step_val, train_loss.loss)
                steps_per_sec = (global_step_val - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step_val
                time_acc = 0

            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2])

            if global_step_val % eval_interval == 0:
                results = metric_utils.eager_compute(
                    eval_metrics,
                    eval_info_metrics,
                    eval_tf_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics',
                    use_function=False,
                )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, global_step_val)
                metric_utils.log_metrics(eval_metrics+eval_info_metrics)

            if global_step_val % train_checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step_val)

            if global_step_val % policy_checkpoint_interval == 0:
                policy_checkpointer.save(global_step=global_step_val)

            if global_step_val % rb_checkpoint_interval == 0:
                rb_checkpointer.save(global_step=global_step_val)

        logging.info('Training finished')
        # tf.profiler.experimental.stop()

        return train_loss


def main(params):
    tf.compat.v1.enable_v2_behavior()
    # tf.debugging.enable_check_numerics()    # error out inf or NaN
    logging.set_verbosity(logging.INFO)
    # gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_c)
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
    encoder_fc_layers = [512, 512]
    actor_fc_layers = [512, 512]
    critic_obs_fc_layers = [512, 512]
    critic_action_fc_layers = [512, 512]
    critic_joint_fc_layers = [512, 512]

    print('==================================================')
    # for k, v in FLAGS.flag_values_dict().items():
    #     print(k, v)
    print('conv_1d_layer_params', conv_1d_layer_params)
    print('conv_2d_layer_params', conv_2d_layer_params)
    print('encoder_fc_layers', encoder_fc_layers)
    print('actor_fc_layers', actor_fc_layers)
    print('critic_obs_fc_layers', critic_obs_fc_layers)
    print('critic_action_fc_layers', critic_action_fc_layers)
    print('critic_joint_fc_layers', critic_joint_fc_layers)
    print('==================================================')

    # HACK: pfnet is not supported with parallel environments currently
    assert not params.use_parallel_envs

    # HACK: supporting only one parallel env currently
    assert params.num_parallel_environments == 1
    assert params.num_parallel_environments_eval == 1

    # init_env_pfnet = FLAGS.init_env_pfnet
    # config_file = FLAGS.config_file
    # action_timestep = FLAGS.action_timestep
    # physics_timestep = FLAGS.physics_timestep
    # is_localize_env = FLAGS.is_localize_env

    # compute observation channel dim
    # if FLAGS.obs_mode == 'rgb-depth':
    #     FLAGS.obs_ch = 4
    # elif FLAGS.obs_mode == 'rgb':
    #     FLAGS.obs_ch = 3
    # elif FLAGS.obs_mode == 'depth' or FLAGS.obs_mode == 'occupancy_grid':
    #     FLAGS.obs_ch = 1
    # else:
    #     raise ValueError

    # set random seeds
    # random.seed(FLAGS.seed)
    # np.random.seed(FLAGS.seed)
    # tf.random.set_seed(FLAGS.seed)

    pfnet_model = init_pfnet_model(params, is_igibson=True)
    model_ids = None
    model_ids_eval = None

    train_eval(
        root_dir=params.root_dir,
        gpu=params.device_idx,
        env_load_fn=lambda model_id, mode, use_tf_function, device_idx: suite_gibson.create_env(params, pfnet_model=pfnet_model, do_wrap_env=True),
        #     config_file=config_file,
        #     model_id=model_id,
        #     env_mode=mode,
        #     use_tf_function=use_tf_function,
        #     init_pfnet=init_env_pfnet,
        #     is_localize_env=is_localize_env,
        #     action_timestep=action_timestep,
        #     physics_timestep=physics_timestep,
        #     device_idx=device_idx,
        # ),
        model_ids=model_ids,
        eval_env_mode=params.env_mode,
        num_iterations=params.num_iterations,
        conv_1d_layer_params=conv_1d_layer_params,
        conv_2d_layer_params=conv_2d_layer_params,
        encoder_fc_layers=encoder_fc_layers,
        actor_fc_layers=actor_fc_layers,
        critic_obs_fc_layers=critic_obs_fc_layers,
        critic_action_fc_layers=critic_action_fc_layers,
        critic_joint_fc_layers=critic_joint_fc_layers,
        initial_collect_steps=params.initial_collect_steps,
        collect_steps_per_iteration=params.collect_steps_per_iteration,
        num_parallel_environments=params.num_parallel_environments,
        replay_buffer_capacity=params.replay_buffer_capacity,
        train_steps_per_iteration=params.train_steps_per_iteration,
        batch_size=params.rl_batch_size,
        actor_learning_rate=params.actor_learning_rate,
        critic_learning_rate=params.critic_learning_rate,
        alpha_learning_rate=params.alpha_learning_rate,
        gamma=params.gamma,
        use_tf_functions=params.use_tf_function,
        use_parallel_envs=params.use_parallel_envs,
        num_eval_episodes=params.num_eval_episodes,
        eval_interval=params.eval_interval,
        eval_only=params.eval_only,
        num_parallel_environments_eval=params.num_parallel_environments_eval,
        model_ids_eval=model_ids_eval,
        debug_summaries=params.debug,
    )


if __name__ == '__main__':
    # flags.mark_flag_as_required('root_dir')
    # flags.mark_flag_as_required('config_file')
    parsed_params = parse_common_args('igibson', add_rl_args=True)

    run_name = Path(parsed_params.root_dir).name
    run = wandb.init(config=parsed_params, name=run_name, project=WANDB_PROJECT, sync_tensorboard=True, mode='disabled' if parsed_params.debug else 'online')

    multiprocessing.handle_main(functools.partial(main, parsed_params))
    # app.run(main)
    main(parsed_params)



# python -u train_eval.py --root_dir 'train_output' --eval_only=False --num_iterations 3000 --initial_collect_steps 500 --use_parallel_envs=True --collect_steps_per_iteration 1 --num_parallel_environments 1 --num_parallel_environments_eval 1 --replay_buffer_capacity 1000 --train_steps_per_iteration 1 --batch_size 16 --num_eval_episodes 10 --eval_interval 500 --device_idx=0 --seed 100 --pfnet_loadpath=/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint
# nohup python -u train_eval.py --root_dir=train_output --eval_only=False --num_iterations=3000 --initial_collect_steps=500 --use_parallel_envs=no --collect_steps_per_iteration=1 --num_parallel_environments=1 --num_parallel_environments_eval=1 --replay_buffer_capacity=1000 --train_steps_per_iteration=1 --rl_batch_size=16 --num_eval_episodes=10 --eval_interval=500 --device_idx=2 --seed=100 --custom_output rgb_obs depth_obs likelihood_map --pfnet_loadpath=/home/honerkam/repos/deep-activate-localization/src/rl_agents/run2/train_navagent/chks/checkpoint_25_0.076/pfnet_checkpoint > nohup_rl.out &
