import numpy as np
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from torch import nn
import torch
import gym
from wandb.integration.sb3 import WandbCallback
import os
import wandb
from pathlib import Path


def out_sz(in_size, k=3, pad=0, stride=1):
    return ((in_size - k + 2 * pad) / stride + 1).astype(int)



def get_torch_encoder(conv_2d_layer_params, encoder_fc_layers, in_shape, activation_fn=nn.ReLU()):
    in_channels = in_shape[2]
    out_shape = np.array(in_shape[:2])
    
    layers = []
    
    if conv_2d_layer_params is not None:
        for (filters, kernel_size, strides) in conv_2d_layer_params:
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides))
            layers.append(activation_fn)
            
            in_channels = filters
            out_shape = out_sz(in_size=out_shape, k=kernel_size, stride=strides)


        layers.append(nn.Flatten())
        in_features = np.prod(list(out_shape) + [in_channels])
        
    if encoder_fc_layers is not None:
        for num_units in encoder_fc_layers:
            layers.append(nn.Linear(in_features=in_features, out_features=num_units))
            layers.append(activation_fn)
            
            in_features = num_units
            
    return nn.Sequential(*layers), in_features
    


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, conv_2d_layer_params, encoder_fc_layers):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        
        
        def _get_2d_enc(key: str, shape=None):
            return get_torch_encoder(conv_2d_layer_params=conv_2d_layer_params,
                                     encoder_fc_layers=encoder_fc_layers,
                                     in_shape=shape or observation_space[key].shape)[0]
        
        preprocessing_layers = {}
        total_concat_size = 0
        
        if 'rgb_obs' in observation_space.keys():
            preprocessing_layers['rgb_obs'] = _get_2d_enc('rgb_obs')
            total_concat_size += encoder_fc_layers[-1]
        if 'depth_obs' in observation_space.keys():
            preprocessing_layers['depth_obs'] = _get_2d_enc('depth_obs')
            total_concat_size += encoder_fc_layers[-1]
        if 'floor_map' in observation_space.keys():
            preprocessing_layers['floor_map'] = _get_2d_enc('floor_map')
            total_concat_size += encoder_fc_layers[-1]
        if 'likelihood_map' in observation_space.keys():
            # if np.max(observation_space['likelihood_map'].shape) > 100:
            #     shape = [50, 50, 1]
            # else: 
            #     shape = observation_space['likelihood_map'].shape
            preprocessing_layers['likelihood_map'] = _get_2d_enc('likelihood_map')
            total_concat_size += encoder_fc_layers[-1]
        if "occupancy_grid" in observation_space.keys():
            preprocessing_layers['occupancy_grid'] = _get_2d_enc('occupancy_grid')
            total_concat_size += encoder_fc_layers[-1]
        if 'scan' in observation_space.keys():
            raise NotImplementedError(observation_space.keys())
            preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=conv_1d_layer_params,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'task_obs' in observation_space.keys():
            # preprocessing_layers['task_obs'] = get_torch_encoder(conv_2d_layer_params=None, 
            #                   encoder_fc_layers=encoder_fc_layers,
            #                   in_shape=observation_space['task_obs'].shape)
            # total_concat_size += encoder_fc_layers[-1]
            preprocessing_layers['task_obs'] = nn.Flatten()
            total_concat_size += observation_space['task_obs'].shape[-1]
        if 'kmeans_cluster' in observation_space.keys():
            preprocessing_layers['kmeans_cluster'] = get_torch_encoder(conv_2d_layer_params=None, 
                                                                       encoder_fc_layers=encoder_fc_layers, 
                                                                       in_shape=observation_space['kmeans_cluster'].shape)[0]
            total_concat_size += encoder_fc_layers[-1]

        self.extractors = nn.ModuleDict(preprocessing_layers)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if len(observations[key].shape) == 4:
                o = observations[key].permute([0, 3, 1, 2])
            else:
                o = observations[key]
            encoded_tensor_list.append(extractor(o))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
    
    
class CustomCombinedExtractor2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, conv_2d_layer_params, encoder_fc_layers):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor2, self).__init__(observation_space, features_dim=1)
        
        
        def _get_2d_enc(key: str, shape=None):
            return get_torch_encoder(conv_2d_layer_params=conv_2d_layer_params,
                                     encoder_fc_layers=encoder_fc_layers,
                                     in_shape=shape or observation_space[key].shape)[0]
        
        preprocessing_layers = {}
        total_concat_size = 0
        
        if ('rgb_obs' in observation_space.keys()) and ('depth_obs' in observation_space.keys()):
            shp = list(observation_space['rgb_obs'].shape[:2]) + [4]
            preprocessing_layers['rgb_depth'] = _get_2d_enc(None, shp)
            total_concat_size += encoder_fc_layers[-1]
        else:
            if 'rgb_obs' in observation_space.keys():
                preprocessing_layers['rgb_obs'] = _get_2d_enc('rgb_obs')
                total_concat_size += encoder_fc_layers[-1]
            if 'depth_obs' in observation_space.keys():
                preprocessing_layers['depth_obs'] = _get_2d_enc('depth_obs')
                total_concat_size += encoder_fc_layers[-1]
        if 'floor_map' in observation_space.keys():
            preprocessing_layers['floor_map'] = _get_2d_enc('floor_map')
            total_concat_size += encoder_fc_layers[-1]
        if 'likelihood_map' in observation_space.keys():
            # if np.max(observation_space['likelihood_map'].shape) > 100:
            #     shape = [50, 50, 1]
            # else: 
            #     shape = observation_space['likelihood_map'].shape
            preprocessing_layers['likelihood_map'] = _get_2d_enc('likelihood_map')
            total_concat_size += encoder_fc_layers[-1]
        if "occupancy_grid" in observation_space.keys():
            preprocessing_layers['occupancy_grid'] = _get_2d_enc('occupancy_grid')
            total_concat_size += encoder_fc_layers[-1]
        if 'scan' in observation_space.keys():
            raise NotImplementedError(observation_space.keys())
            preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=conv_1d_layer_params,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'task_obs' in observation_space.keys():
            # preprocessing_layers['task_obs'] = get_torch_encoder(conv_2d_layer_params=None, 
            #                   encoder_fc_layers=encoder_fc_layers,
            #                   in_shape=observation_space['task_obs'].shape)
            # total_concat_size += encoder_fc_layers[-1]
            preprocessing_layers['task_obs'] = nn.Flatten()
            total_concat_size += observation_space['task_obs'].shape[-1]
        if 'kmeans_cluster' in observation_space.keys():
            preprocessing_layers['kmeans_cluster'] = get_torch_encoder(conv_2d_layer_params=None, 
                                                                       encoder_fc_layers=encoder_fc_layers, 
                                                                       in_shape=observation_space['kmeans_cluster'].shape)[0]
            total_concat_size += encoder_fc_layers[-1]

        self.extractors = nn.ModuleDict(preprocessing_layers)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == 'rgb_depth':
                d = observations['depth_obs'].permute([0, 3, 1, 2])
                rgb = observations['rgb_obs'].permute([0, 3, 1, 2])
                o = torch.cat([rgb, d], dim=1)
            else:
                if len(observations[key].shape) == 4:
                    o = observations[key].permute([0, 3, 1, 2])
                else:
                    o = observations[key]
            encoded_tensor_list.append(extractor(o))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


    
class CustomCombinedExtractor3(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, conv_2d_layer_params, encoder_fc_layers):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor3, self).__init__(observation_space, features_dim=1)
        
        
        def _get_2d_enc(key: str, shape=None):
            return get_torch_encoder(conv_2d_layer_params=conv_2d_layer_params,
                                     encoder_fc_layers=encoder_fc_layers,
                                     in_shape=shape or observation_space[key].shape)
        
        preprocessing_layers = {}
        total_concat_size = 0
        
        if ('rgb_obs' in observation_space.keys()) and ('depth_obs' in observation_space.keys()):
            shp = list(observation_space['rgb_obs'].shape[:2]) + [4]
            preprocessing_layers['rgb_depth'], out_features = _get_2d_enc(None, shp)
            total_concat_size += out_features
        else:
            if 'rgb_obs' in observation_space.keys():
                preprocessing_layers['rgb_obs'], out_features = _get_2d_enc('rgb_obs')
                total_concat_size += out_features
            if 'depth_obs' in observation_space.keys():
                preprocessing_layers['depth_obs'], out_features = _get_2d_enc('depth_obs')
                total_concat_size += out_features
        if 'floor_map' in observation_space.keys():
            preprocessing_layers['floor_map'], out_features = _get_2d_enc('floor_map')
            total_concat_size += out_features
        if 'likelihood_map' in observation_space.keys():
            # if np.max(observation_space['likelihood_map'].shape) > 100:
            #     shape = [50, 50, 1]
            # else: 
            #     shape = observation_space['likelihood_map'].shape
            preprocessing_layers['likelihood_map'], out_features = _get_2d_enc('likelihood_map')
            total_concat_size += out_features
        if "occupancy_grid" in observation_space.keys():
            preprocessing_layers['occupancy_grid'], out_features = _get_2d_enc('occupancy_grid')
            total_concat_size += out_features
        if 'scan' in observation_space.keys():
            raise NotImplementedError(observation_space.keys())
            preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=conv_1d_layer_params,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'task_obs' in observation_space.keys():
            # preprocessing_layers['task_obs'] = get_torch_encoder(conv_2d_layer_params=None, 
            #                   encoder_fc_layers=encoder_fc_layers,
            #                   in_shape=observation_space['task_obs'].shape)
            # total_concat_size += encoder_fc_layers[-1]
            preprocessing_layers['task_obs'] = nn.Flatten()
            total_concat_size += observation_space['task_obs'].shape[-1]
        if 'kmeans_cluster' in observation_space.keys():
            preprocessing_layers['kmeans_cluster'], out_features = get_torch_encoder(conv_2d_layer_params=None, 
                                                                       encoder_fc_layers=encoder_fc_layers, 
                                                                       in_shape=observation_space['kmeans_cluster'].shape)
            total_concat_size += out_features

        self.extractors = nn.ModuleDict(preprocessing_layers)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == 'rgb_depth':
                d = observations['depth_obs'].permute([0, 3, 1, 2])
                rgb = observations['rgb_obs'].permute([0, 3, 1, 2])
                o = torch.cat([rgb, d], dim=1)
            else:
                if len(observations[key].shape) == 4:
                    o = observations[key].permute([0, 3, 1, 2])
                else:
                    o = observations[key]
            encoded_tensor_list.append(extractor(o))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
    
    
class MyWandbCallback(WandbCallback):      
    def save_model(self) -> None:
        super().save_model()
        
        step_path = os.path.join(self.model_save_path, f"model_step{self.num_timesteps}.zip")
        self.model.save(step_path)
        wandb.save(step_path, base_path=self.model_save_path)


def get_run_name(params):
    return f"{params.agent}_{params.obs_mode}_p{params.num_particles}{params.init_particles_distr}rng{params.particles_range}std{params.init_particles_std[0]}_r{params.resample}a{params.alpha_resample_ratio}_t{np.round(params.transition_std[0], 3)}_{np.round(params.transition_std[1], 3)}"


def get_logdir(run_name: str):
    project_root = Path(__file__).parent.parent.parent.parent
    
    i = 0
    logdir = project_root / 'logs' / f'{run_name}'
    while logdir.exists():
        logdir = project_root / 'logs' / f'{run_name}{i}'
        i += 1
    logdir.mkdir(parents=True)
    return logdir

    
class DotDict(dict):
    """
    Source: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    Example:
    m = DotDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


class MetricsCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # one info for each env in vectorized env
        infos = self.locals["infos"]
        for i, info in enumerate(infos):
            self.logger.record_mean('rollout/collision_penalty', info['collision_penalty'])
            self.logger.record_mean('rollout/pred', float(info['pred']))
            self.logger.record_mean('rollout/reward', float(info['reward']))
            if info['done']:
                self.logger.record_mean('rollout/success', info['success'])
                self.logger.record_mean('rollout/path_length', info['path_length'])
                self.logger.record_mean('rollout/pred_final', float(info['pred']))
                # self.logger.record_mean('rollout/collision_per_eps', self.training_env.envs[i].collision_step)
        return True