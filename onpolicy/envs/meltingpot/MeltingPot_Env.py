# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#

"""Wraps a meltingpot environment to be used as a dm_env environment """

import os, sys
from typing import Tuple, Any, Mapping, Callable, Dict, List, Optional, Union, NamedTuple
import dm_env
import dmlab2d
import gymnasium as gym
from matplotlib import pyplot as plt
from gymnasium import spaces
from ml_collections import config_dict
import numpy as np
from ray.rllib.env import multi_agent_env
import tree
from gym.vector import VectorEnv
from ray import cloudpickle
from ray.util.iter import ParallelIteratorWorker
from collections.abc import Mapping, Sequence
import cv2

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
sys.path.append(parent_dir)
from meltingpot import substrate as meltingpot_substrate
from meltingpot.utils.substrates.wrappers import observables
from meltingpot.utils.substrates import substrate

PLAYER_STR_FORMAT = 'player_{index}'
_WORLD_PREFIX = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']
MAX_CYCLES = 400

_OBSERVATION_PREFIX = ['WORLD.RGB', 'RGB', 'ORIENTATION']


def timestep_to_observations(timestep: dm_env.TimeStep) -> Mapping[str, Any]:
    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value
            for key, value in observation.items()
            if key in _OBSERVATION_PREFIX
        }
    return gym_observations


def remove_world_observations_from_space(
        observation: spaces.Dict) -> spaces.Dict:
    return spaces.Dict({
        key: observation[key] for key in observation if key not in _WORLD_PREFIX
    })


def spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
    """Converts a dm_env nested structure of specs to a Gym Space.

    BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
    Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

    Args:
      spec: The nested structure of specs

    Returns:
      The Gym space corresponding to the given spec.
    """
    if isinstance(spec, dm_env.specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.floating):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        elif np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        else:
            raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
    elif isinstance(spec, (list, tuple)):
        return spaces.Tuple([spec_to_space(s) for s in spec])
    elif isinstance(spec, dict):
        return spaces.Dict({key: spec_to_space(s) for key, s in spec.items()})
    else:
        raise ValueError('Unexpected spec of type {}: {}'.format(type(spec), spec))


###


# plotting WORLD.RGB images

class DataExtractor:
    def __init__(self, data):
        self.data = data

    def extract_world_rgb(self):
        """
        Extracts 'WORLD.RGB' arrays from the data.
        """

        return [item['WORLD.RGB'] for item in self.data]

    def plot_and_save_rgb_images(self):
        """
        Plots and saves the extracted 'WORLD.RGB' arrays.
        """
        # Create a folder named "plot" if it doesn't exist.
        if not os.path.exists("plot"):
            os.mkdir("plot")

        # Extract WORLD.RGB values.
        world_rgbs = self.extract_world_rgb()

        # Loop through the extracted RGB arrays, plot them, and save them.
        for i, rgb in enumerate(world_rgbs):
            if isinstance(rgb, np.ndarray):
                plt.imshow(rgb)
                plt.title(f'World RGB Image {i + 1}')
                plt.axis('off')  # Do not show axis in the plot
                filename = os.path.join("plot", f"world_rgb_{i + 1}.png")
                plt.savefig(filename)
                plt.close()  # Close the plot to avoid showing it while running the code
            else:
                raise TypeError("The RGB data is not in the correct numpy array format.")


# Example data


###
class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
    """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

    def __init__(self, env: dmlab2d.Environment, max_cycles: int = MAX_CYCLES):
        """Initializes the instance.

        Args:
          env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
        """
        self._env = env
        self._num_players = len(self._env.observation_spec())

        self._ordered_agent_ids = [
            PLAYER_STR_FORMAT.format(index=index)
            for index in range(self._num_players)
        ]
        # RLLib requires environments to have the following member variables:
        # observation_space, action_space, and _agent_ids
        self._agent_ids = set(self._ordered_agent_ids)
        # RLLib expects a dictionary of agent_id to observation or action,
        # Melting Pot uses a tuple, so we convert
        self.observation_space = self._convert_spaces_tuple_to_dict(
            spec_to_space(self._env.observation_spec()),
            remove_world_observations=True)
        self.action_space = self._convert_spaces_tuple_to_dict(
            spec_to_space(self._env.action_spec()))

        self.share_observation_space = self._create_world_rgb_observation_space(
            self._env.observation_spec()
        )
        # territory room share observation Box(0, 255, (168, 168, 3), uint8)

        # ts=self._env.reset()
        # extractor = DataExtractor(ts.observation)
        # extractor.plot_and_save_rgb_images()

        self.max_cycles = max_cycles
        self.num_cycles = 0

        super().__init__()

    def reset(self, *args, **kwargs):
        """See base class."""
        timestep = self._env.reset()
        self.num_cycles = 0
        return timestep_to_observations(timestep), {}

    def step(self, action_dict):
        """See base class."""

        def is_iterable(obj):
            try:
                iter(obj)
                return True
            except TypeError:
                return False

        action_dict = action_dict if is_iterable(action_dict[0]) else [[item] for item in action_dict]

        if len(action_dict) == 1:
            action_dict = action_dict[0]

        actions = [list(map(int, action_dict[agent_id])) for agent_id, player in enumerate(self._ordered_agent_ids)]
        actions = np.array(actions)

        # Initialize empty arrays to store rewards and done flags for each agent
        agent_rewards = {agent_id: [] for agent_id in self._ordered_agent_ids}
        agent_dones = {agent_id: [] for agent_id in self._ordered_agent_ids}
        agent_observation = {agent_id: {'RGB': [], 'WORLD.RGB': []} for agent_id in self._ordered_agent_ids}
        # Loop through each time step
        for i in range(actions.shape[1]):
            # Step the environment for each agent individually

            timestep = self._env.step(actions[:, i])
            for agent_id, player in enumerate(self._ordered_agent_ids):
                # Append rewards and done flags for each agent
                agent_rewards[player].append(timestep.reward[agent_id])
                agent_dones[player].append(timestep.last())
                obs = timestep_to_observations(timestep)
                agent_observation[player]['RGB'].append(obs[player]['RGB'])
                agent_observation[player]['WORLD.RGB'].append(obs[player]['WORLD.RGB'])
                # Check if the maximum number of cycles is reached
                truncation = self.num_cycles >= self.max_cycles
                agent_dones[player][-1] = agent_dones[player][-1] or truncation

        # Extract the final rewards and done flags for each agent
        rewards = {agent_id: np.array(reward_list) for agent_id, reward_list in agent_rewards.items()}
        done = {agent_id: np.array(done_list, dtype=bool) for agent_id, done_list in agent_dones.items()}
        observations = {}
        for agent_id in self._ordered_agent_ids:
            observations[agent_id] = {
                'RGB': np.stack(agent_observation[agent_id]['RGB'], axis=0),
                'WORLD.RGB': np.stack(agent_observation[agent_id]['WORLD.RGB'], axis=0)
            }

        # (n_rollout, 11, 11, 3), (n_rollout, 30, 21, 3)
        info = {}
        self.num_cycles += 1
        return observations, rewards, done, info

    def close(self):
        """See base class."""
        self._env.close()

    def get_dmlab2d_env(self):
        """Returns the underlying DM Lab2D environment."""
        return self._env

    # Metadata is required by the gym `Env` class that we are extending, to show
    # which modes the `render` method supports.
    metadata = {'render.modes': ['rgb_array']}

    def render(self) -> np.ndarray:
        """Render the environment.

        This allows you to set `record_env` in your training config, to record
        videos of gameplay.

        Returns:
            np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
            representing RGB values for an x-by-y pixel image, suitable for turning
            into a video.
        """
        observation = self._env.observation()
        world_rgb = observation[0]['WORLD.RGB']

        # RGB mode is used for recording videos
        return world_rgb

    def _convert_spaces_tuple_to_dict(
            self,
            input_tuple: spaces.Tuple,
            remove_world_observations: bool = False) -> spaces.Dict:
        """Returns spaces tuple converted to a dictionary.

        Args:
          input_tuple: tuple to convert.
          remove_world_observations: If True will remove non-player observations.
        """
        return spaces.Dict({
            agent_id: (remove_world_observations_from_space(input_tuple[i])
                       if remove_world_observations else input_tuple[i])
            for i, agent_id in enumerate(self._ordered_agent_ids)
        })

    def _create_world_rgb_observation_space(self, observation_spec):
        """
          Creates a space for 'WORLD.RGB' observations for each player.

          Args:
              observation_spec: A nested structure defining the observation space
                                for the environment.

          Returns:
              A Dict space containing the 'WORLD.RGB' observation space for each
              player.
        """
        # Extract 'WORLD.RGB' specs and convert them to Gym spaces
        world_rgb_spec = [
            player_obs_spec['WORLD.RGB']
            for player_obs_spec in observation_spec
        ]

        world_rgb_space = spaces.Tuple([
            spec_to_space(spec) for spec in world_rgb_spec
        ])

        # Map agent ids to their respective 'WORLD.RGB' observation space
        return spaces.Dict({
            agent_id: world_rgb_space[i]
            for i, agent_id in enumerate(self._ordered_agent_ids)
        })


def downsample_observation(array: np.ndarray, scaled) -> np.ndarray:
    """Downsample image component of the observation.
    Args:
      array: RGB array of the observation provided by substrate
      scaled: Scale factor by which to downsaple the observation
    
    Returns:
      ndarray: downsampled observation  
    """

    frame = cv2.resize(
        array, (array.shape[0] // scaled, array.shape[1] // scaled), interpolation=cv2.INTER_AREA)
    return frame





def _downsample_multi_spec(spec, scaled):
    return dm_env.specs.Array(shape=(spec.shape[0] // scaled, spec.shape[1] // scaled, spec.shape[2]), dtype=spec.dtype)


class DownSamplingSubstrateWrapper(observables.ObservableLab2dWrapper):
    """Downsamples 8x8 sprites returned by substrate to 1x1. 
    
    This related to the observation window of each agent and will lead to observation RGB shape to reduce
    from [88, 88, 3] to [11, 11, 3]. Other downsampling scales are allowed but not tested. Thsi will lead
    to significant speedups in training.
    """

    def __init__(self, substrate: substrate.Substrate, cfg):
        super().__init__(substrate)
        self.cfg = cfg
        self.agent_scale = cfg["agent_scale"]
        self.world_scale = cfg["world_scale"]

    def reset(self) -> dm_env.TimeStep:
        timestep = super().reset()
        return self.down_sample_multi_timestep(timestep)

    def step(self, actions) -> dm_env.TimeStep:
        timestep = super().step(actions)
        return self.down_sample_multi_timestep(timestep)

    def observation_spec(self) -> Sequence[Mapping[str, Any]]:
        spec = super().observation_spec()
        res = []
        for s in spec:
            tmp = {}
            for k, v in s.items():
                if k == 'WORLD.RGB':
                    tmp[k] = _downsample_multi_spec(v, self.world_scale)
                elif k == 'RGB':
                    tmp[k] = _downsample_multi_spec(v, self.agent_scale)
                else:
                    tmp[k] = v
            res.append(tmp)
        return res

    def down_sample_multi_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        new_obs = []
        for observation in timestep.observation:
            tmp = {}
            for k, v in observation.items():
                if k == 'WORLD.RGB':
                    tmp[k] = downsample_observation(v, self.world_scale)
                elif k == 'RGB':
                    tmp[k] = downsample_observation(v, self.agent_scale)
                else:
                    tmp[k] = v
            new_obs.append(tmp)

        return timestep._replace(observation=new_obs)


def env_creator(env_config):
    """Outputs an environment for registering."""
    env_config = config_dict.ConfigDict(env_config)
    env = meltingpot_substrate.build(env_config['substrate'], roles=env_config['roles'])

    env = DownSamplingSubstrateWrapper(env, env_config)

    env = MeltingPotEnv(env)

    return env
