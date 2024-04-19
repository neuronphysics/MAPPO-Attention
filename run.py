import random
import os
import warnings


import comet_ml
import supersuit as ss
import numpy as np

from gym import spaces

import torch
from torch.nn import Module

import hydra
from omegaconf import DictConfig

from src.envs import get_env
from src.envs import ObstoStateWrapper, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1, black_death_v3, PermuteObsWrapper, AddStateSpaceActMaskWrapper, CooperativeRewardsWrapper, ParallelEnv
from src.replay_buffer import ReplayBuffer, ReplayBufferImageObs


# MeltingPot imports
from src.envs.meltingpot.meltingpot import substrate
from src.envs.MeltingPot_Env import env_creator
from src.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from src.envs.separated_buffer import SeparatedReplayBuffer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse

class MeltingPotArgs:
    def __init__(self):
        self.episode_length = 1000
        self.n_rollout_threads = 1
        self.hidden_size = 180
        self.recurrent_N = 1
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.use_gae = True
        self.use_popart = False
        self.use_valuenorm = True
        self.use_proper_time_limits = False

def make_train_env(env_config):
    
    if env_config.family == 'marlgrid':
        envs = [AddStateSpaceActMaskWrapper(PermuteObsWrapper(CooperativeRewardsWrapper(get_env(env_config.name, env_config.family, env_config.params)))) for _ in range(env_config.rollout_threads)]
        env = ParallelEnv(envs)
        return env
    
    elif env_config.family == 'meltingpot':

        def env_fn():
            rank=0
            player_roles = substrate.get_config('territory__rooms').default_player_roles
            scale_factor = 8
            env_config_dict = {"substrate": 'territory__rooms', "roles": player_roles, "scaled": scale_factor}
            env = env_creator(env_config_dict)
            env.reset(0 + rank * 1000)
            return env
        melt_env = DummyVecEnv([env_fn]) 
        return melt_env
    
    env_class = get_env(env_config.name, env_config.family, env_config.params)
    env = env_class.parallel_env(**env_config.params)
    
    if env_config.continuous_action:
        env = ss.clip_actions_v0(env)
    if env_config.family != 'starcraft':
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
    else:
        env = black_death_v3(env)

    env = ObstoStateWrapper(env)
    
    if env_config.family == 'starcraft':
        env = pettingzoo_env_to_vec_env_v1(env, black_death=True)
    else:
        env = pettingzoo_env_to_vec_env_v1(env, black_death=False)
    env = concat_vec_envs_v1(env, env_config.rollout_threads, num_cpus=1, base_class='gym')

    return env

def make_eval_env(env_config):
    
    if env_config.family == 'marlgrid':
        envs = [AddStateSpaceActMaskWrapper(PermuteObsWrapper(get_env(env_config.name, env_config.family, env_config.params))) for _ in range(1)]
        env = ParallelEnv(envs)
        return env

    elif env_config.family == 'meltingpot':


        def env_fn():
            rank=0
            player_roles = substrate.get_config('territory__rooms').default_player_roles
            scale_factor = 8
            env_config_dict = {"substrate": 'territory__rooms', "roles": player_roles, "scaled": scale_factor}
            env = env_creator(env_config_dict)
            env.reset(50000 + rank * 10000)
            return env
        melt_env = DummyVecEnv([env_fn])  # Pass a list of functions
        return melt_env
    
    
    env_class = get_env(env_config.name, env_config.family, env_config.params)

    env = env_class.parallel_env(**env_config.params)
    
    if env_config.continuous_action:
        env = ss.clip_actions_v0(env)
    if env_config.family != 'starcraft':
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
    else:
        env = black_death_v3(env)
    env = ObstoStateWrapper(env)

    return env

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    
    cfg.env.family = 'meltingpot' if cfg.env.name == 'meltingpot' else cfg.env.family
    
    train_envs = make_train_env(cfg.env)
    eval_env = make_eval_env(cfg.env)

    if cfg.env.family == 'meltingpot':
        print(dir(train_envs))
        print(train_envs.share_observation_space["player_0"].shape)
        print(train_envs.observation_space["player_0"].shape)
        print(train_envs.action_space["player_0"].shape)

        
    if cfg.env.family=='meltingpot':
        size_x , size_y , size_z = train_envs.share_observation_space["player_0"].shape
        observation_space = train_envs.observation_space["player_0"]['RGB']
        shared_observation_space = train_envs.share_observation_space["player_0"]
        #observation_space=np.reshape(observation_space, (z , x ,y ))
        print(observation_space.shape)
        #observation_space = np.moveaxis(observation_space, -1, 0)

        action_space = train_envs.action_space["player_0"]

        if cfg.env.obs_type == 'image' and cfg.policy.params.type == 'conv':
            state_space = spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(cfg.policy.params.conv_out_size * cfg.n_agents,),
                dtype='float',
            )
        elif isinstance(train_envs.state_space, tuple):
            state_space = train_envs.state_space[0]
        else:
            state_space = train_envs.state_space

    else:
        if isinstance(train_envs.observation_space, spaces.Dict):
            observation_space = train_envs.observation_space['observation']
        elif isinstance(train_envs.observation_space, tuple):
            observation_space = train_envs.observation_space[0]
        else:
            observation_space = train_envs.observation_space

        if isinstance(train_envs.action_space, tuple):
            action_space = train_envs.action_space[0]
        else:
            action_space = train_envs.action_space

        if cfg.env.obs_type == 'image' and cfg.policy.params.type == 'conv':
            state_space = spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(cfg.policy.params.conv_out_size * cfg.n_agents,),
                dtype='float',
            )
        elif isinstance(train_envs.state_space, tuple):
            state_space = train_envs.state_space[0]
        else:
            state_space = train_envs.state_space

    print(f"Observation Space: {observation_space.shape} | Action Space: {action_space.shape} | State Space: {state_space.shape}")

    print("POLICY")
    policy = hydra.utils.instantiate(
        cfg.policy, 
        observation_space=observation_space, 
        action_space=action_space, 
        state_space=state_space, 
        params=cfg.policy.params)
    
    print("BUFFER")
    policy = policy.to(device)
    

    if cfg.env.family == 'meltingpot':
        args_meltingpot = MeltingPotArgs()
        buffer = SeparatedReplayBuffer(args_meltingpot , observation_space , shared_observation_space , action_space)

    else:

        if cfg.env.obs_type == 'image' and cfg.policy.params.type == 'conv':
            buffer = ReplayBufferImageObs(observation_space, action_space, cfg.buffer, device)
        else:
            buffer = ReplayBuffer(observation_space, action_space, state_space, cfg.buffer, device)
        
    print(dir(buffer))
    print("RUNNER")
    runner = hydra.utils.instantiate(
        cfg.runner,
        train_env=train_envs,
        eval_env=eval_env,
        env_family=cfg.env.family,
        policy=policy, 
        buffer=buffer, 
        params=cfg.runner.params, 
        device=device)
    
    if not cfg.test_mode:
        runner.run()

    mean_rewards, std_rewards, mean_wins, std_wins = runner.evaluate()
    print(f"Eval Rewards: {mean_rewards} +- {std_rewards} | Eval Win Rate: {mean_wins} +- {std_wins}")
    train_envs.close()
    eval_env.close()

if __name__ == "__main__":
    main()
    