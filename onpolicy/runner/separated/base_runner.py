import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.all_args.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
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
        for agent_id in range(self.num_agents):
            if not self.env_name == "Meltingpot":
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                    self.envs.observation_space[agent_id]
                po = Policy(self.all_args,
                            self.envs.observation_space[agent_id],
                            share_observation_space,
                            self.envs.action_space[agent_id],
                            device=self.device)
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
                            device=self.device)
                # policy network

            self.policy.append(po)
        
        ##count total number of parameters
        print(f"total number of parameters of this model is {self.count_parameters()}")
        if self.model_dir is None:
            self.model_dir = self._find_model_dir(self.job_id)
            if self.model_dir is not None and self.all_args.load_model:
                self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
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
            self.trainer[agent_id].prep_rollout()
            if self.all_args.rnn_attention_module == "LSTM":
                 next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                       self.buffer[agent_id].rnn_states_critic[-1],
                                                                       self.buffer[agent_id].rnn_cells_critic[-1],
                                                                       self.buffer[agent_id].masks[-1])
            else:
                next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                       self.buffer[agent_id].rnn_states_critic[-1],
                                                                       self.buffer[agent_id].masks[-1])            
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        # random update order
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in torch.randperm(self.num_agents):
            tmp_buf = self.buffer[agent_id]
            self.trainer[agent_id].prep_training()
            tmp_buf.update_factor(factor)
            available_actions = None if tmp_buf.available_actions is None \
                else tmp_buf.available_actions[:-1].reshape(-1, *tmp_buf.available_actions.shape[2:])

            obs = tmp_buf.obs[:-1].reshape(-1, *tmp_buf.obs.shape[2:])
            rnn_states = tmp_buf.rnn_states[0:1].reshape(-1, *tmp_buf.rnn_states.shape[2:])
            
            actions = tmp_buf.actions.reshape(-1, *tmp_buf.actions.shape[2:])
            masks = tmp_buf.masks[:-1].reshape(-1, *tmp_buf.masks.shape[2:])
            active_masks = tmp_buf.active_masks[:-1].reshape(-1, *tmp_buf.active_masks.shape[2:])
            if self.all_args.rnn_attention_module == "LSTM":
                
                rnn_cells = tmp_buf.rnn_cells[0:1].reshape(-1, *tmp_buf.rnn_cells.shape[2:])
                old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(obs,
                                                                                              rnn_states,
                                                                                              rnn_cells,
                                                                                              actions,
                                                                                              masks,
                                                                                              available_actions,
                                                                                              active_masks)
                train_info = self.trainer[agent_id].train(tmp_buf)
                new_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(obs,
                                                                                              rnn_states,
                                                                                              rnn_cells,
                                                                                              actions,
                                                                                              masks,
                                                                                              available_actions,
                                                                                              active_masks)
            else:
                old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(obs,
                                                                                             rnn_states,
                                                                                             actions,
                                                                                             masks,
                                                                                             available_actions,
                                                                                             active_masks)
            

                train_info = self.trainer[agent_id].train(tmp_buf)

                new_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(obs,
                                                                                              rnn_states,
                                                                                              actions,
                                                                                              masks,
                                                                                              available_actions,
                                                                                              active_masks)

            factor = factor * _t2n(
                torch.prod(torch.exp(new_actions_logprob - old_actions_logprob), dim=-1).reshape(self.episode_length,
                                                                                                 self.n_rollout_threads,
                                                                                                 1))

            train_infos.append(train_info)

            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(),  os.path.join(self.save_dir, f"actor_agent_{agent_id}.pt"))
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(),  os.path.join(self.save_dir, f"critic_agent_{agent_id}.pt"))
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), os.path.join(self.save_dir, f"vnrom_agent_{agent_id}.pt"))

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(os.path.join(self.model_dir, f'actor_agent_{agent_id}.pt'))
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(os.path.join(self.model_dir, f'critic_agent_{agent_id}.pt'))
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom_state_dict = torch.load(os.path.join(self.model_dir, f'vnrom_agent_{agent_id}.pt'))
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

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
            actor_parameters += sum(p.numel() for p in self.policy[agent_id].actor.parameters())
            critic_parameters += sum(p.numel() for p in self.policy[agent_id].critic.parameters())
        return actor_parameters + critic_parameters
