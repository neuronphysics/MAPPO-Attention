import math

import torch
import torch.nn as nn

from onpolicy.algorithms.utils.lstm import LSTMLayer
from onpolicy.algorithms.utils.util import print_trainable_parameters, init, check, ObsDataset, selectively_unfreeze_layers
from onpolicy.algorithms.utils.cnn import CNNBase, Encoder
from onpolicy.algorithms.utils.modularity import SCOFF
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.rim_cell import RIM
from absl import logging
from onpolicy.algorithms.utils.QSA.train_qsa import generate_model, cosine_anneal
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, TaskType


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(R_Actor, self).__init__()

        # new parameters
        self.args = args
        self.drop_out = args.drop_out
        self.rnn_attention_module = args.rnn_attention_module
        self.use_bidirectional = args.use_bidirectional
        self.n_rollout = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self._use_version_scoff = args.use_version_scoff
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy

        obs_shape = get_shape_from_obs_space(obs_space)
        # improve the speed
        self.accumulation_steps = 4  # Adjust this based on your needs

        self.use_attention = args.use_attention
        self._attention_module = args.attention_module
        self.use_slot_att = args.use_slot_att

        self._obs_shape = obs_shape
        self.global_step = 0

        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_slot_att:
            model = generate_model(args)
            print(model.state_dict().keys())

            list_modules = ["slot_attn.slot_attention.project_q", "slot_attn.slot_attention.project_k", "slot_attn.slot_attention.project_v", "slot_proj", "out"]

            if args.fine_tuning_type =='Lora':
                # Define the LoRA configuration
                lora_config = LoraConfig(
                    r=16,  # Rank of the low-rank update
                    lora_alpha=32,  # Scaling factor
                    lora_dropout=0.1,  # Dropout probability
                    target_modules=list_modules,  # Target specific layers
                    bias="none",
                    modules_to_save=None,    # Don't save any modules fully
                    task_type= TaskType.FEATURE_EXTRACTION,  # Optimized for feature extraction
                )
                # Apply LoRA to the selected layers of the SlotAttention module
                self.slot_attn = get_peft_model(model, lora_config).to(device)
                print_trainable_parameters(self.slot_attn)  # check the fraction of parameters trained
                for n, p in self.slot_attn.model.named_parameters():
                    if 'lora' in n:
                        print(f"New parameter {n:<13} | {p.numel():>5} parameters | updated")
            elif args.fine_tuning_type == "Partial":
                selectively_unfreeze_layers(model, list_modules)
                self.slot_attn =  model.to(device)

            self.tau = args.tau_start
            self.sigma = args.sigma_start
            args.use_input_att = False
            args.use_x_reshape = True

        if self.use_attention and len(self._obs_shape) >= 3:
            if self._attention_module == "RIM":
                self.rnn = RIM(device, self.hidden_size, self.hidden_size // args.rim_num_units, args.rim_num_units,
                               args.rim_topk, args)
            elif self._attention_module == "SCOFF":
                self.rnn = SCOFF(device, self.hidden_size, self.hidden_size, args.scoff_num_units, args.scoff_topk,
                                 args)
        elif not self.use_attention:
            if len(obs_shape) == 3:
                logging.info('Not using any attention module, input width: %d ', obs_shape[1])
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                if self.rnn_attention_module == "GRU":
                    self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
                else:
                    self.rnn = LSTMLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, rnn_cells, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_cells = check(rnn_cells).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.use_slot_att:
            # slot att model takes (batch, 3, H, W) and returns a dict
            batch, _, _, _ = obs.shape

            # your slot attention or other GPU-intensive tasks
            actor_features = self.slot_attn(obs.permute(0, 3, 1, 2), tau=self.tau, sigma=self.sigma, is_Train=False,
                                            visualize=False)['slots'].reshape(batch, -1)

        else:
            actor_features = self.base(obs)

        if self.use_attention:
            output = self.rnn(actor_features, rnn_states, rnn_cells, masks=masks)
        else:
            if self.rnn_attention_module == "GRU":
                output = self.rnn(actor_features, rnn_states, masks=masks)
            else:
                x, (h, c) = self.rnn(actor_features, rnn_states, rnn_cells, masks=masks)
                output = (x, h)

        # expect actor_feature (batch, input_size) rnn_state (1, batch, hidden_size)
        actor_features, rnn_states = output[:2]
        if self.use_attention and self.rnn_attention_module == "LSTM":
            rnn_cells = output[-1]
        else:
            rnn_cells = rnn_cells.permute(1, 0, 2)

        if not self.use_attention and (self._use_naive_recurrent_policy or self._use_recurrent_policy):
            rnn_states = rnn_states.permute(1, 0, 2)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states, rnn_cells

    def evaluate_actions(self, obs, rnn_states, rnn_cells, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        if rnn_cells is not None:
            rnn_cells = check(rnn_cells).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.use_slot_att:
            # Process in chunks to avoid OOM
            # slot att model takes (batch, 3, H, W) and returns a dict
            batch_size = obs.size(0)
            chunk_size = self.args.slot_pretrain_batch_size
            features_list = []
            total_processed = 0
        
            # Use torch.cuda.amp for mixed precision training
            with torch.cuda.amp.autocast():
                for i in range(0, batch_size, chunk_size):
                    end_idx = min(i + chunk_size, batch_size)  # Ensure we don't exceed batch size
                    chunk = obs[i:end_idx]
                    chunk_batch_size = chunk.size(0)  # Get actual size of current chunk
                
                    with torch.no_grad():
                        chunk_features = self.slot_attn(
                                        chunk.permute(0, 3, 1, 2),
                                        tau=self.tau,
                                        sigma=self.sigma
                        )["slots"]
                        features_list.append(chunk_features)
                    
                    total_processed += chunk_batch_size
                
                # Verify we processed the entire batch
                assert total_processed == batch_size, f"Processed {total_processed} samples but batch size is {batch_size}"
                                    
                # Concatenate all chunks
                features = torch.cat(features_list, dim=0)
                actor_features = features.reshape(batch_size, -1)

        else:
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy or self.use_attention:

            if self.use_attention:
                output = self.rnn(actor_features, rnn_states, rnn_cells, masks=masks)
            else:
                if self.rnn_attention_module == "GRU":
                    output = self.rnn(actor_features, rnn_states, masks=masks)
                else:
                    x, (h, c) = self.rnn(actor_features, rnn_states, rnn_cells, masks=masks)
                    output = (x, h)

        actor_features, rnn_states = output[:2]

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.act.evaluate_actions_trpo(
                actor_features,
                action, available_actions,
                active_masks=
                active_masks if self._use_policy_active_masks
                else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                       action, available_actions,
                                                                       active_masks=
                                                                       active_masks if self._use_policy_active_masks
                                                                       else None)

        return action_log_probs, dist_entropy

    def train_slot_att(self, obs, cur_ppo_idx, optimizer, scheduler):
        """
        if not dist.is_initialized():
           rank, world_size, local_rank, distributed_device = distributed_setup()
        else:
              rank, world_size, local_rank, distributed_device = dist.get_rank(), dist.get_world_size(), dist.get_local_rank(), torch.device(f"cuda:{local_rank}")
        """
        self.global_step = self.args.ep_counter.get_cur_ep() * self.args.ppo_epoch + cur_ppo_idx

        self.tau = cosine_anneal(self.global_step, self.args.tau_steps, start_value=self.args.tau_start,
                                 final_value=self.args.tau_final)
        self.sigma = cosine_anneal(self.global_step, self.args.sigma_steps, start_value=self.args.sigma_start,
                                   final_value=self.args.sigma_final)

        # slot att model takes (batch, 3, H, W) and returns a dict
        # TODO: we shouldnt move the slot attention module, just for now, lets keep it on GPU 1 (second gpu)

        # ddp_model=DDP(self.slot_att, device_ids=[local_rank], output_device=local_rank)
        logging.debug("Starting training slot attention module from checkpoints and on multiple gpus.")

        # Create a DistributedSampler with shuffle=False
        obs_dataset = ObsDataset(obs)
        # sampler = torch.utils.data.distributed.DistributedSampler(obs_dataset,num_replicas=world_size, rank=rank, shuffle=False)

        # Create a DataLoader with the DistributedSampler
        dataloader = torch.utils.data.DataLoader(obs_dataset,
                                                batch_size=self.args.slot_pretrain_batch_size // self.accumulation_steps,
                                                num_workers=4,
                                                pin_memory=True,
                                                persistent_workers=True
                                                 )

        slot_att_total_loss = 0

        # Iterate over the dataloader
        for obs_minibatch in dataloader:
            # Move the minibatch to the GPU
            obs_minibatch = obs_minibatch.permute(0, 3, 1, 2).to(**self.tpdv)
            optimizer.zero_grad()
            # Forward pass through the slot attention model
            out_tmp = self.slot_attn(obs_minibatch, tau=self.tau, sigma=self.sigma, is_Train=True, visualize=False)
            accum_adjustment = len(obs_minibatch) / len(dataloader.dataset)
            accum_consistency_encoder_loss = (
                    out_tmp['loss']['compositional_consistency_loss'].item() * accum_adjustment
            )
            # Compute the loss
            minibatch_loss = out_tmp['loss']['mse'] + out_tmp['sim_loss'] + out_tmp['loss']['cross_entropy']
            if self.args.use_consistency_loss:
                minibatch_loss += accum_consistency_encoder_loss
            # Normalize the loss to account for accumulation
            minibatch_loss.backward()
            # Perform optimizer step
            optimizer.step()

            # Clip gradients 
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.slot_attn.parameters()),
                                     self.args.slot_clip_grade_norm)

            # Step the scheduler
            scheduler.step(self.global_step)
            slot_att_total_loss += minibatch_loss.detach().item()
            # Accumulate the loss            

        return slot_att_total_loss  # Scale the loss back up for reporting


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(R_Critic, self).__init__()

        # new parameters
        self.drop_out = args.drop_out
        self.rnn_attention_module = args.rnn_attention_module
        self.use_bidirectional = args.use_bidirectional
        self.n_rollout = args.n_rollout_threads
        self.scoff_do_relational_memory = args.scoff_do_relational_memory

        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy

        ## Zahra added
        self._use_version_scoff = args.use_version_scoff
        self.use_attention = args.use_attention
        self._attention_module = args.attention_module

        self.scoff_num_modules_read_input = args.scoff_num_modules_read_input
        self.scoff_inp_heads = args.scoff_inp_heads
        self.scoff_share_comm = args.scoff_share_comm
        self.scoff_share_inp = args.scoff_share_inp
        self.scoff_memory_mlp = args.scoff_memory_mlp
        self.scoff_memory_slots = args.scoff_memory_slots
        self.scoff_memory_head_size = args.scoff_memory_head_size
        self.scoff_num_memory_heads = args.scoff_num_memory_heads
        self.scoff_num_memory_topk = args.scoff_memory_topk

        if args.use_slot_att and not args.use_input_att:
            args.use_input_att = True

        self._obs_shape = cent_obs_shape

        base = CNNBase if len(self._obs_shape) == 3 else MLPBase
        self.base = base(args, self._obs_shape)

        if self.use_attention and len(self._obs_shape) >= 3:

            if self._attention_module == "RIM":
                self.rnn = RIM(device, self.hidden_size, self.hidden_size // args.rim_num_units, args.rim_num_units,
                               args.rim_topk, args)

            elif self._attention_module == "SCOFF":
                self.rnn = SCOFF(device, self.hidden_size, self.hidden_size, args.scoff_num_units, args.scoff_topk,
                                 args)
        elif not self.use_attention:
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                if self.rnn_attention_module == "GRU":
                    self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
                else:
                    self.rnn = LSTMLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, rnn_cells, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        if rnn_cells is not None:
            rnn_cells = check(rnn_cells).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)

        if self.use_attention:
            output = self.rnn(critic_features, rnn_states, rnn_cells, masks=masks)
        else:
            if self.rnn_attention_module == "GRU":
                output = self.rnn(critic_features, rnn_states, masks=masks)
            else:
                x, (h, c) = self.rnn(critic_features, rnn_states, rnn_cells, masks=masks)
                output = (x, h)

        critic_features, rnn_states = output[:2]
        if self.use_attention and self.rnn_attention_module == "LSTM":
            rnn_cells = output[-1]
        else:
            if rnn_cells is not None:
                rnn_cells = rnn_cells.permute(1, 0, 2)

        if not self.use_attention and (self._use_naive_recurrent_policy or self._use_recurrent_policy):
            rnn_states = rnn_states.permute(1, 0, 2)

        critic_features = critic_features.unsqueeze(0)
        values = self.v_out(critic_features)

        return values, rnn_states, rnn_cells
