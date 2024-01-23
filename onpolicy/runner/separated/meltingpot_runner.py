    
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


def flatten_lists(input_list):
    # Check if input is a list
    if not isinstance(input_list, list):
        raise ValueError("Input is not a list")

    # Check if each element of the list is also a list
    for inner_list in input_list:
        if not isinstance(inner_list, list):
        
           # Check if each element of the inner list is a numpy array
           for item in inner_list:
               if not isinstance(item, np.ndarray):
                   raise ValueError("Inner list does not contain only numpy arrays")

    # Convert to a list of concatenated numpy arrays
    concatenated_arrays = [np.concatenate(inner_list) for inner_list in input_list]
    return concatenated_arrays

class MeltingpotRunner(Runner):
    def __init__(self, config):
        super(MeltingpotRunner, self).__init__(config)
       
    def run(self):
        print('executing warmup function')
        self.warmup()   

        start = time.time()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        print('num episodes to run (separated):', episodes) 

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)
                print(f"After envs.step {step} in MeltingpotRunner (separate) for action size {actions.shape}, the reward {rewards} dones {dones}")

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
                    
                    #JUAN ADDED, OVERALL AVERAGE EPISODE REWARDS
                    total_average_episode_rewards = 0

                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        # Existing code to calculate individual rewards
                        print(f"rewards after run {rewards} here")
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
                print(f"finish log training")
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        #if --n_rollout_threads 6 && --substrate_name "territory__rooms"
        obs = self.envs.reset()
        # replay buffer
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
                              rgb_player.append(item[player]['WORLD.RGB'])
                              arrays.append(item[player]['RGB'])
                              #player_i obs: (11, 11, 3), share_obs: (168, 168, 3)
                    result = np.stack(arrays)
                    image  = np.stack(rgb_player)
            share_obs.append(image)
            agent_obs.append(result)
        share_obs = np.array(share_obs)
        agent_obs = np.array(agent_obs)
        #share_obs shape: (6, 9, 168, 168, 3), agent obs shape: (6, 9, 11, 11, 3)
        for agent_id in range(self.num_agents):
            #size of buffer share_obs (6, 168, 168, 3)--- obs (6, 11, 11, 3)
            self.buffer[agent_id].share_obs[0] = share_obs[:,agent_id,:,:,:].transpose(0, 2, 1, 3).copy()
            self.buffer[agent_id].obs[0]       = agent_obs[:,agent_id,:,:,:].copy()
        

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
            value, action, action_log_prob, rnn_state, rnn_state_critic \
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
            # print(f"meltingpot runner in collect - action log prob shape : {action_log_prob.shape} and action shape {action.shape}")
            if self.envs.action_space[player].__class__.__name__ == 'MultiDiscrete':
                print(f"meltingpot_runner action type {self.envs.action_space[player].__class__.__name__}")
                for i in range(self.envs.action_space[player].shape):
                    uc_action_env = np.eye(self.envs.action_space[player].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[player].__class__.__name__ == 'Discrete':
                # print(f"meltingpot_runner action type {self.envs.action_space[player].__class__.__name__}")
                # print(f"size of action in meltingpot runner {np.eye(self.envs.action_space[player].n)[action].shape}")
                
                var = np.eye(self.envs.action_space[player].n)[action]
                
                action_env = np.squeeze(var, axis=next((axis for axis, size in enumerate(var.shape) if size == 1), None))

            else:
                raise NotImplementedError

            
            # print(f"size of action in the collect function {action.shape}, rnn_state (tuple) {rnn_state[0].shape} rnn_state_critic {rnn_state_critic[0].shape}")
            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state[0]))
            rnn_states_critic.append( _t2n(rnn_state_critic[0]))
            

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)
        
        #print("BEFORE SQUEEZING \n values are \n", np.array(values).shape, "\n actions are \n", np.array(actions).shape, "\n action log probs are \n ", np.array(action_log_probs).shape, " \n rnn states are \n", np.array(rnn_states).shape, "\n rnn states critic are \n", np.array(rnn_states_critic).shape, "\n actions env are \n", len(actions_env))
        values = np.array(values) if isinstance(values, list) else values
        actions = np.array(actions) if isinstance(actions, list) else actions
        action_log_probs = np.array(action_log_probs) if isinstance(action_log_probs, list) else action_log_probs
        rnn_states = np.array(rnn_states) if isinstance(rnn_states, list) else rnn_states
        rnn_states_critic = np.array(rnn_states_critic) if isinstance(rnn_states_critic, list) else rnn_states_critic

        values = values.squeeze(-1).transpose(1, 0, 2)
        if actions.ndim==3:
            actions = actions.transpose(2, 0, 1)
        else:
            actions = actions.squeeze(-1).transpose(1, 0, 2)
        if action_log_probs.ndim==2:
           action_log_probs = action_log_probs[:, np.newaxis, :]
        action_log_probs = action_log_probs.transpose(1, 0, 2)
        if rnn_states.ndim==3:
            rnn_states =rnn_states[:, np.newaxis, :, :]
        rnn_states = rnn_states.transpose(1, 0, 2, 3)
        #print(f"size of rnn state critic {rnn_states_critic.shape}")
        if rnn_states_critic.ndim==3:
            rnn_states_critic = rnn_states_critic[:, np.newaxis, :, :]
        rnn_states_critic = rnn_states_critic.transpose(1, 0, 2, 3)
        #print("AFTER SQUEEZING\n values are \n", values.shape, "\n actions are \n", actions.shape, "\n action log probs are \n ", action_log_probs.shape, " \n rnn states are \n", rnn_states.shape, "\n rnn states critic are \n", rnn_states_critic.shape, "\n length of actions env \n", len(actions_env))
        #print(f"size of values {values.shape} size of actions {actions.shape} size of action log probs {action_log_probs.shape} size of rnn states {rnn_states.shape} size of rnn states critic {rnn_states_critic.shape}")
        #values (1, num_agent, n_rollout)
        #ctions (1, num_agent, n_rollout)
        #rnn states (1, num_agent, n_rollout, hidden_size)
        #rnn_states_critic (1, num_agent, n_rollout, hidden_size)
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        
        obs, rewards, done, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        print('rnn states sep', rnn_states.shape)

        # Extract the boolean values for each player and convert to a boolean array
        done_new  = np.array([player_dict[f'player_{i}'] for player_dict in done for i in range(self.num_agents)], dtype=np.bool_)
        rewards = np.array([player_dict[f'player_{i}'] for player_dict in rewards for i in range(self.num_agents)], dtype=np.float32)
        #rnn_states:(1, num_agent, n_rollout_threads, hidden_size)
        # print(f"done_new shape {done_new.shape}, rewards shape {rewards.shape}")
        done_new = np.expand_dims(done_new, axis=0)
        rewards  = np.expand_dims(rewards, axis=0)
        # Create a boolean mask with the same shape as rnn_states
        
        rnn_states[done_new == True] = np.zeros(((done_new == True).sum(), self.hidden_size), dtype=np.float32)

        
        rnn_states_critic[done_new == True] = np.zeros(((done_new == True).sum(), self.hidden_size), dtype=np.float32)
        
        masks = np.ones(( 1, self.num_agents, self.n_rollout_threads, 1), dtype=np.float32)
        masks[done_new == True] = np.zeros(((done_new == True).sum(), 1), dtype=np.float32)
        
        share_obs = []
        agent_obs = []
        
        for sublist in obs:
            if isinstance(sublist, dict):
               # If sublist is a dictionary, process it directly
               for agent_id in range(self.num_agents):
                   player = f"player_{agent_id}"
                   if player in sublist:
                      share_obs.append(sublist[player]['WORLD.RGB'])
                      agent_obs.append(sublist[player]['RGB'])
            else:
                #print(f"sublist is a list of length {len(sublist)} {sublist}")
                # If sublist is a list, iterate over its elements
                for agent_id in range(self.num_agents):
                    player = f"player_{agent_id}"
                    if player in sublist[0]:
                        share_obs.append(sublist[0][player]['WORLD.RGB'])
                        agent_obs.append(sublist[0][player]['RGB'])

        # print(f" share observation shape {np.array(share_obs).shape}, agent observation shape {np.array(agent_obs).shape}")
        share_obs = np.array(share_obs)
        agent_obs = np.array(agent_obs)
        if share_obs.ndim == 5 and agent_obs.ndim == 5:
           share_obs = share_obs.transpose(1, 0, 3, 2, 4)
           agent_obs = agent_obs.transpose(1, 0, 3, 2, 4)
        else:
           share_obs = np.reshape(share_obs, (1,) + share_obs.shape)
           agent_obs = np.reshape(agent_obs, (1,) + agent_obs.shape)
        
        # print(f"share_obs shape {share_obs.shape}, agent_obs shape {agent_obs.shape}, rewards shape {rewards.shape}, masks shape {masks.shape} values shape {values.shape} actions shape {actions.shape} action_log_probs shape {action_log_probs.shape} rnn_states shape {rnn_states.shape} rnn_states_critic shape {rnn_states_critic.shape}")
        for agent_id in range(self.num_agents):
            
            #For a quick fix to see if the issue is just about the share_obs reshaping, I comment out the conditional reshaping:
            #if not self.use_centralized_V:
            #    share_obs = np.array(list(obs[:, agent_id]))
            #print(f"share_obs {share_obs.shape} again")
            
            self.buffer[agent_id].insert(share_obs[:, agent_id],
                                         agent_obs[:, agent_id],
                                         rnn_states[:, agent_id].swapaxes( 1, 0),
                                         rnn_states_critic[:, agent_id].swapaxes( 1, 0),
                                         actions[:, agent_id].swapaxes( 1, 0),
                                         action_log_probs[:, agent_id].swapaxes( 1, 0),
                                         values[:, agent_id].swapaxes( 1, 0),
                                         rewards[:, agent_id].swapaxes( 1, 0),
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
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            # print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            #obs = self.envs.reset()
            obs = self.envs.reset()[:, 0]
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array', has_mode=False)[0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    player= f"player_{agent_id}"
                    # if not self.use_centralized_V:
                    #     #share_obs = np.array(list(obs[:, agent_id]))
                    #     share_obs=np.array(list(np.expand_dims(obs[0][player]['WORLD.RGB'], axis=0)))
                    #     #share_obs = np.array(list(np.expand_dims(obs[0][player]['RGB'], axis=0)))
                    self.trainer[agent_id].prep_rollout()
                    #action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                    #                                                    rnn_states[:, agent_id],
                    #                                                    masks[:, agent_id],
                    #                                                    deterministic=True)
                    rgb_data = obs[0][player]['RGB'] if isinstance(obs, np.ndarray) else obs[player]['RGB']
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(np.expand_dims(rgb_data, axis=0))),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)
                    action = action.detach().cpu().numpy()
                    # rearrange action
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

                    temp_actions_env.append(action_env)
                    if isinstance(rnn_state, tuple):
                        rnn_states[:, agent_id] = _t2n(rnn_state[0])
                    else:
                        rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                actions_env = np.array(actions_env[0]).swapaxes(0, 1) #Armin
                obs, rewards, dones, infos = self.envs.step(actions_env)
                #Armin: start
                obs = obs[0]
                # print(f"render reward shape {rewards}")
                rewards = rewards[0]
                summed_rewards = []
                for agent_id in range(self.num_agents):
                    summed_rewards.append(np.sum(rewards[player]))
                summed_rewards = np.array(summed_rewards)
                #Armin: end
                episode_rewards.append(rewards)

                
                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                #rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                #masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    #image = self.envs.render('rgb_array')[0][0]
                    image = self.envs.render('rgb_array', has_mode=False)[0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
            player_rewards = {f"player_{agent_id}": [] for agent_id in range(self.num_agents)}

            # Accumulate rewards for each player
            for episode in episode_rewards:
                for agent_id in range(self.num_agents):
                    player = f"player_{agent_id}"
                    player_rewards[player].append(episode[player])

            # Calculate the average reward for each player
            average_episode_rewards = {}
            for agent_id in range(self.num_agents):
                player = f"player_{agent_id}"
                total_rewards = np.sum(player_rewards[player], axis=0)
                average_episode_rewards[player] = np.mean(total_rewards)
                
            # Print the average rewards
            for player, avg_reward in average_episode_rewards.items():
                print(f"eval average episode rewards of {player}: {avg_reward}")
            
            #episode_rewards = np.array(episode_rewards)
            #for agent_id in range(self.num_agents):
            #    average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
            #    print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
