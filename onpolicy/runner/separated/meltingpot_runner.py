    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import cv2
from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()


def flatten_list(data):
    # Check if the first element of data is a list containing numpy arrays
    if isinstance(data[0], list) and all(isinstance(arr, np.ndarray) for arr in data[0]):
        a = []
        for j in range(len(data[0])):
           r = []
           for i in range(len(data)):
               r.append(data[i][j])
           a.append(np.array(r).squeeze())
        return a
    
    # Check if all elements of data are numpy arrays
    elif all(isinstance(arr, np.ndarray) for arr in data):
        return data
    
class MeltingpotRunner(Runner):
    def __init__(self, config):
        super(MeltingpotRunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                #actions_env = flatten_list(actions_env)
                print(f"meltingpot runner separate ......")
                print(f"actions of meltingpot  {actions_env}")    
                # Obser reward and next obs
                obs, rewards, dones, truncations, infos = self.envs.step(actions_env)
                print(f"still in run {obs}")
                data = obs, rewards, dones, truncations, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
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
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for index in self.envs._ordered_agent_ids:
                            idv_rews.append(rewards[index])
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})

                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                        print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        
        if self.env_name=="Meltingpot":
            
            #'RGB': Box(0, 255, (40, 40, 3), uint8))
            share_obs = []
            agent_obs = []
            for sublist in obs:
                for item in sublist:
                    if item:
                        rgb_player=[]
                        arrays = []
                        for agent_id in range(self.num_agents):
                           player= f"player_{agent_id}"
                           if player in item:
                              rgb_player.append(item[player]['RGB'])
                              arrays.append(item[player]['RGB'])
                              img_size= item[player]['RGB'].shape[1]
                        result = np.stack(arrays)
                        image= np.concatenate(rgb_player, axis=0)
                        #print(f"shared observation {image.shape} ")
                        #resize (256,80,40,3) into shape (256,40,40,3) to make it compatible with the code
                        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                        share_obs.append(image)
                        agent_obs.append(result)
                        #print(f"shared observation resized {image.shape} {result.shape}")
            share_obs = np.array(share_obs)
            
        else:
            share_obs = []
            for o in obs:
                share_obs.append(list(chain(*o)))
            share_obs = np.array(share_obs)
        #print(f"size of shared observation {share_obs.shape}")
        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            if self.env_name=="Meltingpot":
                #print(f"agent obs type: {agent_obs}")
                #print(f"agent obsevation {np.array(agent_obs)[:, agent_id].shape}")
                self.buffer[agent_id].obs[0] = np.array(agent_obs)[:, agent_id].copy() #(256, 40, 40, 3)
            else:
               self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_states_actor, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            #value: torch.tensor, action: torch.tensor, action_log_prob :torch.tensor, rnn_states_actor: tuple(torch.tensor, torch.tensor), rnn_state_critic: tuple(torch.tensor, torch.tensor)
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            player= f"player_{agent_id}"
            
            if self.envs.action_space[player].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[player].shape):
                    uc_action_env = np.eye(self.envs.action_space[player].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[player].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[player].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            
            rnn_states.append({f"player_{agent_id}": _t2n(tensor) for agent_id, tensor in enumerate(rnn_states_actor)})
            rnn_states_critic.append({f"player_{agent_id}": _t2n(tensor) for agent_id, tensor in enumerate(rnn_state_critic)})

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)
        
        values = np.array([v.squeeze(-1) for v in values]).transpose(1, 0, 2)
        actions = np.array([a.squeeze(-1) for a in actions]).transpose(1, 0, 2)
        # values (256, 2, 1) actions (256, 2, 1)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        #action log probability (256, 2, 1)
        #rnn_states (64, 9, 1) rnn_states_critic (64, 9, 1)
       
        rnn_states ={key: np.concatenate([d[key] for d in rnn_states]).transpose(1, 0, 2) for key in rnn_states[0]}
        
        rnn_states_critic={key: np.concatenate([d[key] for d in rnn_states_critic]).transpose(1, 0, 2) for key in rnn_states_critic[0]}
        #change for debug
        
        #actions_dict = {}
        #for agent_id in range(self.num_agents):
        #    agent_actions = [actions_pair[agent_id] for actions_pair in actions_env]
        #    stacked_actions = np.vstack(agent_actions)
        #    actions_dict[f"player_{agent_id}"] = stacked_actions
        #action_env = actions_dict
        #actions_dict shape : (256, 8)
        #removed
        #rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        #rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)
        #rnn state shape (1, 2, 64, 2) rnn state of critic (1, 2, 64, 2)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        
        obs, rewards, done, truncations, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[done == True] = np.zeros(((done == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[done == True] = np.zeros(((done == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[done == True] = np.zeros(((done == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_truncations, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

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

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
