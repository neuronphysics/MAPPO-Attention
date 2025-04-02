import wandb
import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.device = torch.device("cuda" if self.all_args.use_cuda and torch.cuda.is_available() else "cpu")
        self.all_args.device = self.device

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

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
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        print('use centralized V', self.use_centralized_V)
        print('obs space', self.envs.observation_space)
        print('action space', )

        share_observation_space = self.envs.share_observation_space['player_0'] if self.use_centralized_V else self.envs.share_observation_space['player_0']

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)
        
        # policy network
        policy = Policy(self.all_args,
                        self.envs.observation_space['player_0']['RGB'],
                        share_observation_space,
                        self.envs.action_space['player_0'],
                        device = self.device)

        # Check for multiple GPUs and wrap the policy
        if self.all_args.use_cuda and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.policy = nn.DataParallel(policy)
            self.policy.to(self.device)
        else:
            self.policy = policy
            self.policy.to(self.device)

        if self.model_dir is not None and self.all_args.load_model:
            self.restore(self.model_dir)

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        else:
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.envs.observation_space['player_0']['RGB'],
                                         share_observation_space,
                                         self.envs.action_space['player_0'])

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
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.rnn_cells_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        
        next_values = np.array(np.split(_t2n(next_values.squeeze(0)), self.n_rollout_threads))

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        policy_to_save = self.policy.module if isinstance(self.policy, nn.DataParallel) else self.policy

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            policy_to_save.save(self.save_dir, episode)
        else:
            policy_actor = policy_to_save.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_{}.pt".format(episode))
            policy_critic = policy_to_save.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_{}.pt".format(episode))

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        policy_to_load = self.policy.module if isinstance(self.policy, nn.DataParallel) else self.policy
        actor_path = str(model_dir) + '/actor.pt'
        critic_path = str(model_dir) + '/critic.pt'

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            policy_to_load.restore(model_dir)
        else:
            # Load actor state dict
            try:
                policy_actor_state_dict = torch.load(actor_path, map_location=self.device)
                policy_to_load.actor.load_state_dict(policy_actor_state_dict)
            except FileNotFoundError:
                print(f"Warning: Actor model file not found at {actor_path}")
            except Exception as e:
                 print(f"Error loading actor state dict from {actor_path}: {e}")

            # Load critic state dict if not just rendering
            if not self.all_args.use_render:
                 try:
                    policy_critic_state_dict = torch.load(critic_path, map_location=self.device)
                    policy_to_load.critic.load_state_dict(policy_critic_state_dict)
                 except FileNotFoundError:
                    print(f"Warning: Critic model file not found at {critic_path}")
                 except Exception as e:
                    print(f"Error loading critic state dict from {critic_path}: {e}")

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                if isinstance(v, float) or isinstance(v, np.float32):
                    res = v
                elif isinstance(v, torch.Tensor):
                    res = v.float()
                else:
                    raise TypeError(f"Unsupported type {type(v)} for value {v}")
                self.writter.add_scalars(k, {k: res}, total_num_steps)


    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)