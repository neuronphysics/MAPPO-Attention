# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#

"""Wraps a meltingpot environment to be used as a dm_env environment """
import os
from typing import  Tuple, Any, Mapping, Callable, Dict, List, Optional, Union, NamedTuple
import dm_env
import dmlab2d
import gymnasium as gym
from matplotlib import pyplot as plt
from gymnasium import spaces
from meltingpot import substrate
from ml_collections import config_dict
import numpy as np
from ray.rllib.env import multi_agent_env
import os
import tree

PLAYER_STR_FORMAT = 'player_{index}'
_WORLD_PREFIX = 'WORLD.'
MAX_CYCLES = 1000

def timestep_to_observations(timestep: dm_env.TimeStep) -> Mapping[str, Any]:
  gym_observations = {}
  for index, observation in enumerate(timestep.observation):
    gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
        key: value
        for key, value in observation.items()
        if _WORLD_PREFIX not in key
    }
  return gym_observations


def remove_world_observations_from_space(
    observation: spaces.Dict) -> spaces.Dict:
  return spaces.Dict({
      key: observation[key] for key in observation if _WORLD_PREFIX not in key
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
    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    rewards = {
        agent_id: timestep.reward[index]
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }
    self.num_cycles += 1
    termination = timestep.last()
    done = { agent_id:termination for agent_id in self._ordered_agent_ids}
    truncation = self.num_cycles >= self.max_cycles
    truncations = {agent_id: truncation for agent_id in self._ordered_agent_ids}
    info = {}

    observations = timestep_to_observations(timestep)
    return observations, rewards, done, truncations, info

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


def env_creator(env_config):
  """Outputs an environment for registering."""
  env_config = config_dict.ConfigDict(env_config)
  env = substrate.build(env_config['substrate'], roles=env_config['roles'])
  env = MeltingPotEnv(env)
  return env


