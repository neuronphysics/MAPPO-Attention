import numpy as np
import os
from os.path import expanduser, expandvars
import time
from datetime import datetime
from pathlib import Path

from gym import spaces

from tqdm import tqdm
import comet_ml

import torch


unique_dir_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
base_save_dir = "/home/juan-david-vargas-mazuera/ICML-RUNS/CODES/saf/scripts/training/outputs"


from src.envs.separated_buffer import SeparatedReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()

class PGRunner():
    def __init__(self, train_env, eval_env, env_family, policy, buffer, params, device):


        self.policy_type = policy.type
        self.total_timesteps = params.total_timesteps
        self.batch_size = params.rollout_threads * params.env_steps
        self.latent_kl = params.latent_kl
        self.rollout_threads = params.rollout_threads
        self.n_agents = params.n_agents
        self.env_steps = params.env_steps
        self.lr_decay = params.lr_decay
        self.env_family = env_family
        self.eval_episodes = params.eval_episodes
        self.use_comet = True if params.comet else False
        self.checkpoint_dir = params.checkpoint_dir
        self.save_dir = Path(base_save_dir) / unique_dir_name

       
       
       #MELTINGPOT  - territory rooms
        self.num_agents = 9
        self.episode_length   = 1000
        self.n_rollout_threads = 1
        self.num_env_steps= 40e6
        
        if isinstance(train_env.observation_space, spaces.Dict):
            self.observation_space = train_env.observation_space['observation']
        elif isinstance(train_env.observation_space, tuple):
            self.observation_space = train_env.observation_space[0]
        else:
            self.observation_space = train_env.observation_space

        if isinstance(train_env.action_space, tuple):
            self.action_space = train_env.action_space[0]
        else:
            self.action_space = train_env.action_space
        
        test_mode = params.test_mode

        self.train_env = train_env
        self.eval_env = eval_env
        self.policy = policy
        self.device = device
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(os.getcwd())


        if self.checkpoint_dir is not None and not test_mode:
            print("Resuming training from", self.checkpoint_dir)
            self.load_checkpoints(self.checkpoint_dir)
            if self.use_comet:
                api_key_path= Path(self.checkpoint_dir)/Path('apy_key.txt')
                
                with open(api_key_path) as f:
                    self.exp_api_key = f.readlines()
                # Check to see if there is a key in environment:
                EXPERIMENT_KEY = os.environ.get("COMET_EXPERIMENT_KEY", self.exp_api_key[0])

                # First, let's see if we continue or start fresh:
                if (EXPERIMENT_KEY is not None):
                    # There is one, but the experiment might not exist yet:
                    api = comet_ml.API() # Assumes API key is set in config/env
                    try:
                        api_experiment = api.get_experiment_by_key(EXPERIMENT_KEY)
                    except Exception:
                        api_experiment = None
                    if api_experiment is not None:
                        CONTINUE_RUN = True
                        # We can get the last details logged here, if logged:
                        self.step = int(api_experiment.get_parameters_summary("curr_step")["valueCurrent"])
                    
        else: 
            if test_mode:
                print("Loading model from", self.checkpoint_dir)
                self.load_checkpoints(self.checkpoint_dir)
            if self.use_comet:
                self.exp = comet_ml.Experiment(project_name=params.comet.project_name)
                self.exp.set_name(params.comet.experiment_name)
                self.exp_key = self.exp.get_key()
                
        #new meltingpot
        self.buffer = []
        self.trainer=[]

        for agent_id in range(self.num_agents):
            tr=self.policy
            
            player_key = f"player_{agent_id}"
            share_observation_space =  self.train_env.share_observation_space[player_key]
            bu = SeparatedReplayBuffer( self.train_env.observation_space[player_key]['RGB'],
                                        share_observation_space,
                                        self.train_env.action_space[player_key])

            self.buffer.append(bu)
            self.trainer.append(tr)
        print("BUFFER", len(self.buffer))


    def env_reset(self, mode="train"):
        '''
        Resets the environment.
        Returns:
        obs: [rollout_threads, n_agents, obs_shape]
        '''
        if mode == "train":
            timestep = self.train_env.reset()
            observations=self.train_env.timestep_to_observations(timestep)
            
        elif mode == "eval":
            timestep = self.eval_env.reset()
            observations=self.eval_env.timestep_to_observations(timestep)

        self.num_cycles = 0

        return observations, {}


    def env_step(self, action, mode="train"):
        '''
        Does a step in the defined environment using action.
        Args:
            action: [rollout_threads, n_agents] for Discrete type and [rollout_threads, n_agents, action_dim] for Box type
        '''

        print(self.action_space.__class__.__name__, "ACTION SPACE")
        if self.action_space.__class__.__name__ == 'Box':
            action_ = action.reshape(-1, action.shape[-1]).cpu().numpy()
        elif self.action_space.__class__.__name__ == 'Discrete':
            action_ = action.reshape(-1).cpu().numpy()
        else:
            raise NotImplementedError
        
        if mode == "train":
            obs,  reward, done, info = self.train_env.step(action_)
        elif mode == "eval":
            obs,  reward, done, info = self.eval_env.step(action_)

        obs = torch.Tensor(obs).reshape((-1, self.n_agents)+obs.shape[1:]).to(self.device) # [rollout_threads, n_agents, obs_shape]
        state = torch.Tensor(state).reshape((-1, self.n_agents)+state.shape[1:]).to(self.device) # [rollout_threads, n_agents, state_shape]
        if type(act_masks) != type(None):
            act_masks = torch.Tensor(act_masks).reshape((-1, self.n_agents)+act_masks.shape[1:]).to(self.device) # [rollout_threads, n_agents, act_shape]
        done = torch.Tensor(done).reshape((-1, self.n_agents)).to(self.device) # [rollout_threads, n_agents]

        reward = torch.Tensor(reward).reshape((-1, self.n_agents)).to(self.device) # [rollout_threads, n_agent]

        return obs, state, act_masks, reward, done, info
    
    
    def warmup(self):
        # reset env
        # if --n_rollout_threads 6 && --substrate_name "territory__rooms"
        obs = self.train_env.reset()
        
        # replay buffer
        share_obs = []
        agent_obs = []
        for sublist in obs:
            for item in sublist:
                if item:
                    rgb_player = []
                    arrays = []
                    for agent_id in range(self.num_agents):
                        player = f"player_{agent_id}"
                        if player in item:
                            rgb_player.append(item[player]['WORLD.RGB'])
                            arrays.append(item[player]['RGB'])
                            # player_i obs: (11, 11, 3), share_obs: (168, 168, 3)
                    result = np.stack(arrays)
                    image = np.stack(rgb_player)
            share_obs.append(image)
            agent_obs.append(result)
        share_obs = np.array(share_obs)
        agent_obs = np.array(agent_obs)
        # share_obs shape: (6, 9, 168, 168, 3), agent obs shape: (6, 9, 11, 11, 3)
        for agent_id in range(self.num_agents):
            print(agent_id)
            # size of buffer share_obs (6, 168, 168, 3)--- obs (6, 11, 11, 3)
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id, :, :, :].transpose(0, 2, 1, 3).copy()
            self.buffer[agent_id].obs[0] = agent_obs[:, agent_id, :, :, :].copy()
            
        combined_states = np.zeros((self.num_agents, *self.buffer[0].rnn_states.shape))
        # Create a new buffer to hold the combined observations for all agents
        combined_obs = np.zeros((self.num_agents, *self.buffer[0].obs.shape))

        # Combine the observations for each agent into the new buffer
        for agent_id in range(self.num_agents):
            combined_obs[agent_id] = self.buffer[agent_id].obs
            combined_states[agent_id] = self.buffer[agent_id].rnn_states

        #shape, example (9,1001,1,1,100)
        combined_states=combined_states.squeeze(2).squeeze(2)
        combined_obs = combined_obs.squeeze(2)
        # Reshape the combined observations to the desired shape
        obs = combined_obs.transpose(1, 0, 3, 4, 2)
        obs = obs.transpose(0, 1, 3, 4, 2)
        
        state=combined_states.transpose(1, 0, 2)
        
        print(obs.shape, "combined obs shape", state.shape, "combined state shape")
        return obs , state , None

            
    def run(self):

        global_step = 0

        obs, state, act_masks = self.warmup()

        obs = torch.tensor(obs)
        state = torch.tensor(state)
        print(obs.shape, state.shape)
        print(type(obs), type(state))
        print(type(obs[0]), type(state[0]))
        print(type(obs[0][0]), type(state[0][0]))
        print(type(obs[0][0][0]))
        print(type(obs[0][0][0][0]))

        start = time.time()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        next_done = torch.zeros((self.rollout_threads, self.n_agents)).to(self.device)

        print('num episodes to run (separated):', episodes)
        best_return = -1e9
        
        
        if self.latent_kl:
            ## old_observation - shifted tensor (the zero-th obs is assumed to be equal to the first one)
            obs_old = obs.clone()
            obs_old[1:] = obs_old.clone()[:-1]

            if self.policy_type =='conv':
                bs = obs_old.shape[0]
                n_ags = obs_old.shape[1]

                obs_old = obs_old.reshape((-1,)+self.policy.obs_shape)
                obs_old = self.policy.conv(obs_old)
                obs_old = obs_old.reshape(bs, n_ags, self.policy.input_shape)
        else:
            obs_old = None
            
        if not os.path.exists(self.save_dir):

                os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
                os.makedirs(self.save_dir, exist_ok = True)
                api_key_path= Path(self.save_dir)/Path('apy_key.txt')
                
                with open(api_key_path, 'w') as f:
                    f.write(self.exp_key)
                    

        for episode in range(episodes):
            obs, state, act_masks = self.warmup()
            if self.lr_decay:
                self.policy.update_lr(episode, episodes)

            total_rewards = 0
            
            for step in range(self.episode_length):
                global_step += self.rollout_threads

                with torch.no_grad():
                    action, logprob, _, value, _ = self.policy.get_action_and_value(obs, state, act_masks, None, obs_old)
                print(action.shape)
                print(type(action), type(logprob), type(value), "ACTION, LOGPROB, VALUE")
                
                next_obs, next_state, next_act_masks, reward, done, info = self.env_step(action)
                print(next_obs.shape, next_state.shape, next_act_masks.shape, reward.shape, done.shape, info, "ENV STEP RUN")

                
                
                

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "Meltingpot":
                    # JUAN ADDED, OVERALL AVERAGE EPISODE REWARDS
                    total_average_episode_rewards = 0

                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        # Existing code to calculate individual rewards

                        for index in list(self.envs.observation_space.keys()):
                            idv_rews.append(rewards[0][index])
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})

                        # Calculate the average episode reward for the current agent
                        average_episode_reward = np.mean(self.buffer[agent_id].rewards) * self.episode_length
                        train_infos[agent_id].update({"average_episode_rewards": average_episode_reward})
                        print("Average episode rewards for agent {} is {}".format(agent_id, average_episode_reward))

                        # Add the average reward of this agent to the total
                        total_average_episode_rewards += average_episode_reward

                    # Calculate the overall average episode reward for all agents
                    overall_average_episode_reward = total_average_episode_rewards / self.num_agents
                    print("Overall average episode reward for all agents:", overall_average_episode_reward)

                self.log_train(train_infos, total_num_steps)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)





    def evaluate(self):
    
        global_step = 0

        obs, state, act_masks = self.env_reset(mode="eval")
        next_done = torch.zeros((1, self.n_agents)).to(self.device)
        
        if self.latent_kl:
            ## old_observation - shifted tensor (the zero-th obs is assumed to be equal to the first one)
            obs_old = obs.clone()
            obs_old[1:] = obs_old.clone()[:-1]

            if self.policy_type =='conv':
                bs = obs_old.shape[0]
                n_ags = obs_old.shape[1]

                obs_old = obs_old.reshape((-1,)+self.policy.obs_shape)
                obs_old = self.policy.conv(obs_old)
                obs_old = obs_old.reshape(bs, n_ags, self.policy.input_shape)
        else:
            obs_old = None

        agg_returns = []
        agg_winrate = []

        for global_step in range(self.eval_episodes):

            total_rewards = 0
            
            nb_games = np.ones(1)
            nb_wins = np.zeros(1)

            for step in range(self.env_steps):

                with torch.no_grad():
                    action, logprob, _, value, _ = self.policy.get_action_and_value(obs, state, act_masks, None, obs_old)

                next_obs, next_state, next_act_masks, reward, done, info = self.env_step(action, mode="eval")
                
                if self.env_family == 'starcraft':
                    total_rewards += reward.max(-1)[0] # (rollout_threads,)
                    # For each rollout, track the number of games player so far and record the wins for finished games
                    for i in range(1):
                        if torch.isin(1, done[i]):
                            nb_games[i] += 1 
                            for agent_info in info[i*self.n_agents:(i+1)*self.n_agents]:
                                if 'battle_won' in agent_info:
                                    nb_wins[i] += int(agent_info['battle_won'])
                                    break
                else:
                    total_rewards += reward.sum(-1) # (rollout_threads,)
                
                obs = next_obs
                state = next_state
                act_masks = next_act_masks
                next_done = done
                if torch.any(done[0]):
                    break

            if self.env_family == 'starcraft':
                total_rewards = total_rewards.cpu()/nb_games
                total_rewards = total_rewards.mean().item()
                episodic_wins = (nb_wins/nb_games).mean()
                print(f"global_step={global_step}, episodic_return={total_rewards}, episodic_win_rate={episodic_wins}")
            else:
                total_rewards = total_rewards.mean().item()
                print(f"global_step={global_step}, episodic_return={total_rewards}")
                if self.use_comet:
                    self.exp.log_metric("episodic_return", total_rewards, global_step)

         
            agg_returns.append(total_rewards)
            if self.env_family == 'starcraft':
                agg_winrate.append(episodic_wins)
            else:
                agg_winrate.append(0)
        
        mean_rewards = np.mean(agg_returns)
        std_rewards = np.std(agg_returns)

        mean_wins = np.mean(agg_winrate)
        std_wins = np.std(agg_winrate)       
        
        return mean_rewards, std_rewards, mean_wins, std_wins


    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            action, logprob, _, value, _  \
                = self.trainer[agent_id].get_action_and_value( self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].masks[step],
                                                            None,
                                                            None)
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            player = f"player_{agent_id}"

            if self.train_env.action_space[player].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.train_env.action_space[player].shape):
                    uc_action_env = np.eye(self.train_env.action_space[player].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.train_env.action_space[player].__class__.__name__ == 'Discrete':
                var = np.eye(self.train_env.action_space[player].n)[action]
                action_env = np.squeeze(var,
                                        axis=next((axis for axis, size in enumerate(var.shape) if size == 1), None))

            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(logprob))

        values = np.array(values) if isinstance(values, list) else values
        actions = np.array(actions) if isinstance(actions, list) else actions
        action_log_probs = np.array(action_log_probs) if isinstance(action_log_probs, list) else action_log_probs

        values = values.squeeze(-1).transpose(1, 0, 2)
        if actions.ndim == 3:
            actions = actions.transpose(2, 0, 1)
        else:
            actions = actions.squeeze(-1).transpose(1, 0, 2)
        if action_log_probs.ndim == 2:
            action_log_probs = action_log_probs[:, np.newaxis, :]
        action_log_probs = action_log_probs.transpose(1, 0, 2)
       
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, None
    
    def load_checkpoints(self, checkpoint_dir):
        self.policy.load_checkpoints(checkpoint_dir)

    def save_checkpoints(self, checkpoint_dir):
        self.policy.save_checkpoints(checkpoint_dir)
            
