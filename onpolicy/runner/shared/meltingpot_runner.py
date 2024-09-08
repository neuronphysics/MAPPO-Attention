import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner
import imageio

def _t2n(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, tuple):
        return tuple(tensor.detach().cpu().numpy() for tensor in x)
    else:
        return x

class MeltingpotRunner(Runner):
    def __init__(self, config):
        super(MeltingpotRunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        print('num episodes to run (shared):', episodes)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

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
                    train_infos["average_episode_rewards"] = np.sum(self.buffer.rewards) / self.buffer.rewards.shape[2]
                    train_infos["performance_score"] = min_max_normalize(train_infos["average_episode_rewards"], 
                                                                         self.all_args.substrate_name)
                    # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                    print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)


    def warmup(self):
        obs = self.envs.reset()

        data_dict = obs[0][0]
        stacked_data = np.stack([np.squeeze(data_dict[f'player_{i}']['RGB']) for i in range(self.num_agents)])
        new_obs = stacked_data[np.newaxis, ...]

        stacked_share_data = np.stack([np.squeeze(data_dict[f'player_{i}']['WORLD.RGB']) for i in range(self.num_agents)])
        new_share_obs = stacked_share_data[np.newaxis, ...]
        new_share_obs = np.transpose(new_share_obs, (0, 1, 3, 2, 4))

        if self.use_centralized_V:
            new_share_obs = new_share_obs.reshape(self.n_rollout_threads, -1)
            new_share_obs = np.expand_dims(new_share_obs, 1).repeat(self.num_agents, axis=1)

        self.buffer.share_obs[0] = new_share_obs.copy()
        self.buffer.obs[0] = new_obs.copy()


    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        
        if self.n_rollout_threads == 1:
            values = np.array(_t2n(value))
            actions = np.array(_t2n(action))
            action_log_probs = np.array(_t2n(action_log_prob))
            rnn_states = np.array(_t2n(rnn_states))
            rnn_states_critic = np.array(_t2n(rnn_states_critic))
        else:
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))


        
        

        # rearrange action
        if self.envs.action_space['player_0'].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space['player_0'].shape):
                uc_actions_env = np.eye(self.envs.action_space['player_0'].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                    
        elif self.envs.action_space['player_0'].__class__.__name__ == 'Discrete':
            
            actions_env = np.squeeze(np.eye(self.envs.action_space['player_0'].n)[actions], 2)

        else:
            raise NotImplementedError

        

        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env


    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        print("before change ", rnn_states.shape, rnn_states_critic.shape)
        data_dict = obs[0]
        stacked_data = np.stack([np.squeeze(data_dict[f'player_{i}']['RGB']) for i in range(self.num_agents)])
        new_obs = stacked_data[np.newaxis, ...]

        stacked_share_data = np.stack([np.squeeze(data_dict[f'player_{i}']['WORLD.RGB']) for i in range(self.num_agents)])
        new_share_obs = stacked_share_data[np.newaxis, ...]
        new_share_obs = np.transpose(new_share_obs, (0, 1, 3, 2, 4))

        rewards = np.array([player_dict[f'player_{i}'] for player_dict in rewards for i in range(self.num_agents)], dtype=np.float32)

        if (dones == True).sum() > 0:
            rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)   
            rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        if (dones == True).sum() > 0:
            masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            new_share_obs = new_share_obs.reshape(self.n_rollout_threads, -1)
            new_share_obs = np.expand_dims(new_share_obs, 1).repeat(self.num_agents, axis=1)

        
        dim3, _, dim1, dim2 = rnn_states.shape
        
        rnn_states = rnn_states.reshape((1, dim1, dim3, dim2))
        rnn_states_critic = rnn_states_critic.reshape((1, dim1, dim3, dim2))

        #rnn_states = np.sum(rnn_states, axis=2, keepdims=True)

        # Do the same for rnn_states_critic if needed
        #rnn_states_critic = np.sum(rnn_states_critic, axis=2, keepdims=True)

        action_log_probs = action_log_probs.reshape((1, dim1, 1))

        self.buffer.insert(new_share_obs, new_obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)


    @torch.no_grad()
    def eval(self,total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_truncations, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)


    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                self.envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.envs.action_space[0].shape):
                        uc_actions_env = np.eye(self.envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, truncations, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)


def min_max_normalize(value, substrate):
    """Return the Elo score for the given substrate. Scores are min-max normalized based on 
    deepmind's reported min score and max score achieved using self-play with a set of different 
    algorithms for the given substrate"""
    minmax_map = {
        'allelopathic_harvest__open': {'min': -17.8, 'max': 92.4},
        'clean_up': {'min': 0.0, 'max': 188.6},
        'prisoners_dilemma_in_the_matrix__arena': {'min': 0.9, 'max': 22.8},
        'territory__rooms': {'min': 10.4, 'max': 236.3}
    }
    return (value - minmax_map[substrate]['min']) / (minmax_map[substrate]['max'] - minmax_map[substrate]['min'])