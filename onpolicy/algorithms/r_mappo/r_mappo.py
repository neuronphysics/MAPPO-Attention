import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check, DormantNeuronTracker
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.use_attention = args.use_attention
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        #entropy related terms
        self.entropy_coef = args.entropy_coef
        self.entropy_initial_coef = args.entropy_coef  # Store the initial value

        self.entropy_final_coef = args.entropy_final_coef
        self.entropy_anneal_duration =args.entropy_anneal_duration
        self.total_updates = 0
        self.warmup_updates = args.warmup_updates
        self.cooldown_updates = args.cooldown_updates
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.use_slot_att = args.use_slot_att
        #
        self.dormant_tracker = DormantNeuronTracker(model=self.policy.actor)
        self.dormant_tracker.initialize_total_neurons()
        self.dormant_tracker.register_activation_hooks()    
        if self.use_attention == True:
            self._use_naive_recurrent_policy = False
            self._use_recurrent_policy = False
        else:
            self._use_naive_recurrent_policy = True
            self._use_recurrent_policy = True

        # assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def update_entropy_coef(self):
        if self.total_updates < self.warmup_updates:
            # During warm-up, keep initial entropy coefficient
            return

        if self.total_updates > (self.entropy_anneal_duration - self.cooldown_updates):
            # During cool-down, keep final entropy coefficient
            self.entropy_coef = self.entropy_final_coef
            return

        # Annealing phase
        progress = (self.total_updates - self.warmup_updates) / (self.entropy_anneal_duration - self.warmup_updates - self.cooldown_updates)
        
        # Cosine annealing schedule
        #The annealing schedule consistently moves from the initial to the final value over the specified duration.
        self.entropy_coef = self.entropy_final_coef + 0.5 * (self.entropy_initial_coef - self.entropy_final_coef) * (1 + np.cos(np.pi * progress))

    

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, idx, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_cells_batch, rnn_states_critic_batch, rnn_cells_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)

        torch.cuda.empty_cache()
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_cells_batch,
                                                                              rnn_states_critic_batch,
                                                                              rnn_cells_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        torch.cuda.empty_cache()
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            
            total_loss = (policy_loss - dist_entropy * self.entropy_coef)

            if self.use_slot_att:
                slot_att_loss= (self.policy.actor.slot_orthoganility_loss + self.policy.actor.slot_consistency_loss)
                total_loss += slot_att_loss
            total_loss.backward()

        actor_parameters =  self.policy.actor.parameters()


       
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(actor_parameters, self.max_grad_norm)
            
        else:
            actor_grad_norm = get_gard_norm(actor_parameters)

        self.policy.actor_optimizer.step()

        # critic update
        torch.cuda.empty_cache()
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        total_critic_loss = (value_loss * self.value_loss_coef)
        total_critic_loss.backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()
        if not self.use_slot_att:
           del actor_parameters
           return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights
        else:
           return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, slot_att_loss


    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if hasattr(torch.cuda, 'memory_reserved'):
           torch.cuda.memory_reserved(0)  # Clear memory reserves

        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['dormant_ratio_actor_net'] = 0
        if self.use_slot_att:
           train_info['slot_att_loss']= 0

        for idx in range(self.ppo_epoch):
            if self._use_recurrent_policy or self.use_attention:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                if not self.use_slot_att:
                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                        = self.ppo_update(idx, sample, update_actor)
                else:   

                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, slot_att_loss \
                        = self.ppo_update(idx, sample, update_actor)
                    train_info['slot_att_loss'] += slot_att_loss


                if (self.total_updates % 50 == 0) and (idx == self.ppo_epoch - 1):

                    train_info['dormant_ratio_actor_net'] += self.dormant_tracker.calculate_dormant_ratio("activation")

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
            
        #update entropy coefficient
        self.update_entropy_coef()
        torch.cuda.empty_cache()

        self.total_updates += 1
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
