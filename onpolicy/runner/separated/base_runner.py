import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        # Set a default device 
        self.all_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_attention = self.all_args.use_attention

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        # Automatically determine job ID and set save directories
        self.job_id = os.environ.get('SLURM_JOB_ID', 'default')

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models' / self.job_id)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = os.path.join(wandb.run.dir, f"Model_ID_{self.job_id}")
                self.run_dir = os.path.join(wandb.run.dir, f"Model_ID_{self.job_id}")
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models' / self.job_id)
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        if self.all_args.algorithm_name == "happo":
            from onpolicy.algorithms.happo.happo_trainer import HAPPO as TrainAlgo
            from onpolicy.algorithms.happo.policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "hatrpo":
            from onpolicy.algorithms.hatrpo.hatrpo_trainer import HATRPO as TrainAlgo
            from onpolicy.algorithms.hatrpo.policy import HATRPO_Policy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        self.policy = []
        num_gpus = torch.cuda.device_count()
        
        for agent_id in range(self.num_agents):
            # Distribute agents across GPUs if multiple are available
            if num_gpus > 1:
                # Assign each agent to a specific GPU based on agent_id
                device = torch.device(f"cuda:{agent_id % num_gpus}")
            else:
                device = self.device
            
            if not self.env_name == "Meltingpot":
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                    self.envs.observation_space[agent_id]
                po = Policy(self.all_args,
                            self.envs.observation_space[agent_id],
                            share_observation_space,
                            self.envs.action_space[agent_id],
                            device=device)
            else:
                player_key = f"player_{agent_id}"
                rgb_shape = self.envs.observation_space[player_key]["RGB"].shape
                sprite_x = rgb_shape[0]
                sprite_y = rgb_shape[1]

                share_observation_space = self.envs.share_observation_space[player_key] if self.use_centralized_V else \
                    self.envs.share_observation_space[player_key]

                po = Policy(self.all_args,
                            self.envs.observation_space[player_key]['RGB'],
                            share_observation_space,
                            self.envs.action_space[player_key],
                            device=device)

            if torch.cuda.device_count() > 1 and self.all_args.use_multi_gpu:
                print(f"Agent {agent_id}: Using {torch.cuda.device_count()} GPUs with DataParallel!")
                policy_wrapped = nn.DataParallel(po)
                policy_wrapped.to(device)
            else:
                print(f"Agent {agent_id}: Using GPU {agent_id % num_gpus if num_gpus > 0 else 'CPU'}")
                policy_wrapped = po
                policy_wrapped.to(device)

            self.policy.append(policy_wrapped)

        ##count total number of parameters
        print(f"total number of parameters of this model is {self.count_parameters()}")
        if self.model_dir is None:
            self.model_dir = self._find_model_dir(self.job_id)
            if self.model_dir is not None and self.all_args.load_model:
                self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # Get the device for this specific agent's policy
            if isinstance(self.policy[agent_id], nn.DataParallel):
                agent_device = self.policy[agent_id].device_ids[0]
                agent_device = torch.device(f'cuda:{agent_device}')
            else:
                agent_device = next(self.policy[agent_id].parameters()).device
            
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=agent_device)
            # buffer
            if not self.env_name == "Meltingpot":
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                    self.envs.observation_space[agent_id]
                bu = SeparatedReplayBuffer(self.all_args,
                                           self.envs.observation_space[agent_id],
                                           share_observation_space,
                                           self.envs.action_space[agent_id])
            else:
                player_key = f"player_{agent_id}"
                share_observation_space = self.envs.share_observation_space[player_key] if self.use_centralized_V else \
                    self.envs.share_observation_space[player_key]
                bu = SeparatedReplayBuffer(self.all_args,
                                           self.envs.observation_space[player_key]['RGB'],
                                           share_observation_space,
                                           self.envs.action_space[player_key])

            self.buffer.append(bu)
            self.trainer.append(tr)

    def _find_model_dir(self, slurm_job_id):
        if self.all_args.use_wandb:
            # Check wandb directory for saved models with the same job ID
            for root, dirs, files in os.walk(wandb.run.dir):
                if f"Model_ID_{slurm_job_id}" in dirs:
                    return os.path.join(root, f"Model_ID_{slurm_job_id}")
        return None

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            # Get the device for this specific agent's policy
            if isinstance(self.policy[agent_id], nn.DataParallel):
                agent_device = self.policy[agent_id].device_ids[0]
            else:
                agent_device = next(self.policy[agent_id].parameters()).device
            
            self.trainer[agent_id].prep_rollout()
            if self.all_args.rnn_attention_module == "LSTM":
                # Move data to the agent's specific device
                share_obs = torch.tensor(self.buffer[agent_id].share_obs[-1]).to(agent_device)
                rnn_states = torch.tensor(self.buffer[agent_id].rnn_states_critic[-1]).to(agent_device)
                rnn_cells = torch.tensor(self.buffer[agent_id].rnn_cells_critic[-1]).to(agent_device)
                masks = torch.tensor(self.buffer[agent_id].masks[-1]).to(agent_device)
                
                next_value = self.trainer[agent_id].policy.get_values(share_obs, rnn_states, rnn_cells, masks)
            else:
                # Move data to the agent's specific device
                share_obs = torch.tensor(self.buffer[agent_id].share_obs[-1]).to(agent_device)
                rnn_states = torch.tensor(self.buffer[agent_id].rnn_states_critic[-1]).to(agent_device)
                masks = torch.tensor(self.buffer[agent_id].masks[-1]).to(agent_device)
                
                next_value = self.trainer[agent_id].policy.get_values(share_obs, rnn_states, None, masks)
            
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in torch.randperm(self.num_agents):
            tmp_buf = self.buffer[agent_id]

            self.trainer[agent_id].prep_training()

            train_info = self.trainer[agent_id].train(tmp_buf)
            train_infos.append(train_info)

            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for agent_id in range(self.num_agents):
            # No need to check for nn.DataParallel here since we're getting the policy_to_save correctly
            policy_to_save = self.policy[agent_id].module if isinstance(self.policy[agent_id], nn.DataParallel) else self.policy[agent_id]

            policy_actor = policy_to_save.actor
            torch.save(policy_actor.state_dict(), os.path.join(self.save_dir, f"actor_agent_{agent_id}.pt"))
            policy_critic = policy_to_save.critic
            torch.save(policy_critic.state_dict(), os.path.join(self.save_dir, f"critic_agent_{agent_id}.pt"))
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), os.path.join(self.save_dir, f"vnrom_agent_{agent_id}.pt"))

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_to_load = self.policy[agent_id].module if isinstance(self.policy[agent_id], nn.DataParallel) else self.policy[agent_id]

            actor_path = os.path.join(self.model_dir, f'actor_agent_{agent_id}.pt')
            critic_path = os.path.join(self.model_dir, f'critic_agent_{agent_id}.pt')
            vnrom_path = os.path.join(self.model_dir, f'vnrom_agent_{agent_id}.pt')

            try:
                policy_actor_state_dict = torch.load(actor_path, map_location=self.device)
                policy_to_load.actor.load_state_dict(policy_actor_state_dict)
            except FileNotFoundError:
                print(f"Warning: Actor model file not found for agent {agent_id} at {actor_path}")
            except Exception as e:
                print(f"Error loading actor state dict for agent {agent_id} from {actor_path}: {e}")

            try:
                policy_critic_state_dict = torch.load(critic_path, map_location=self.device)
                policy_to_load.critic.load_state_dict(policy_critic_state_dict)
            except FileNotFoundError:
                print(f"Warning: Critic model file not found for agent {agent_id} at {critic_path}")
            except Exception as e:
                print(f"Error loading critic state dict for agent {agent_id} from {critic_path}: {e}")

            if self.trainer[agent_id]._use_valuenorm:
                try:
                    policy_vnrom_state_dict = torch.load(vnrom_path, map_location=self.device)
                    self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)
                except FileNotFoundError:
                    print(f"Warning: Value normalizer file not found for agent {agent_id} at {vnrom_path}")
                except Exception as e:
                    print(f"Error loading value normalizer state dict for agent {agent_id} from {vnrom_path}: {e}")

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    #self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
                    if isinstance(v, float) or isinstance(v, np.float32):
                        res = v
                    elif isinstance(v, torch.Tensor):
                        res = v.float()
                    else:
                        raise TypeError(f"Unsupported type {type(v)} for value {v}")
                    self.writter.add_scalars(agent_k, {agent_k: res}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def count_parameters(self):
        actor_parameters = 0
        critic_parameters = 0
        for agent_id in range(self.num_agents):
            if isinstance(self.policy[agent_id], nn.DataParallel):
                # For DataParallel wrapped models, access through the .module attribute
                actor_parameters += sum(p.numel() for p in self.policy[agent_id].module.actor.parameters())
                critic_parameters += sum(p.numel() for p in self.policy[agent_id].module.critic.parameters())
            else:
                actor_parameters += sum(p.numel() for p in self.policy[agent_id].actor.parameters())
                critic_parameters += sum(p.numel() for p in self.policy[agent_id].critic.parameters())
        return actor_parameters + critic_parameters
