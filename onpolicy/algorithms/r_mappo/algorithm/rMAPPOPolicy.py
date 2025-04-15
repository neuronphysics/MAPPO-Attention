import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.QSA.train_qsa import configure_optimizers
from onpolicy.algorithms.utils.util import get_optimizer_groups, WeightClipping
from peft import LoraConfig, get_peft_model
class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.use_slot_att = args.use_slot_att
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.unfreeze_episode = args.unfreeze_episode
        self.unfrozen = False
        
        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        # actor_parameters = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        # critic_parameters = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        self.actor_optimizer = WeightClipping(
                                            get_optimizer_groups(self.actor, args),
                                            beta=args.weight_clip_beta,
                                            optimizer=torch.optim.Adam,
                                            eps=self.opti_eps,
                                            weight_decay=self.weight_decay
                                             )
        """
        self.actor_optimizer = torch.optim.Adam(get_optimizer_groups(self.actor, args),
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        """
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
    
        self._store_initial_weights()
        
    def _store_initial_weights(self):
        """Store initial weights only for LoRA or trainable parameters"""
        self.initial_weights = {}
        
        # For slot attention
        if self.use_slot_att:
            for name, param in self.actor.slot_attn.named_parameters():
                # Only store LoRA or explicitly unfrozen parameters
                if 'lora_' in name or param.requires_grad:
                    self.initial_weights[f"slot_attn.{name}"] = param.detach().clone().requires_grad_(False)

    def check_and_unfreeze(self, current_episode):
        """Check if it's time to unfreeze layers and do so if needed"""
        if not self.unfrozen and current_episode >= self.unfreeze_episode and self.use_slot_att:
            print(f"Episode {current_episode}: Unfreezing slot attention layers")
        
            # Store the current optimizer state before unfreezing
            old_state = None
            if hasattr(self.actor_optimizer, 'optimizer'):
               old_state = self.actor_optimizer.optimizer.state_dict()
        
            # Unfreeze the layers
            #selectively_unfreeze_layers(self.actor.slot_attn, self.actor._finetuned_list_modules)
            # Define LoRA config
            lora_config = LoraConfig(
                                     r=16,  # Rank of the low-rank update
                                     lora_alpha=16,  # Scaling factor
                                     lora_dropout=0.1,
                                     use_rslora=False,
                                     use_dora=True,
                                     target_modules=self.actor._finetuned_list_modules,
                                     init_lora_weights="gaussian",
                                     bias="none",
                                    )
        
            # Convert to LoRA model
            self.actor.slot_attn = get_peft_model(
                                                  self.actor.slot_attn, 
                                                  lora_config
                                                  ).to(self.device)
        
        
            # Create new optimizer
            self.actor_optimizer = WeightClipping(
                                                 get_optimizer_groups(self.actor, self.args),
                                                 beta=self.args.weight_clip_beta,
                                                 optimizer=torch.optim.Adam,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay
                                                 )
        
            # If we had an old state, try to restore compatible parts
            if old_state is not None:
               # Create a new state dict for the new optimizer
               new_state = self.actor_optimizer.optimizer.state_dict()
            
               # Transfer parameter states for parameters that existed in both optimizers
               for param_id in old_state['state']:
                   if param_id in new_state['state']:
                      new_state['state'][param_id] = old_state['state'][param_id]
            
               # Load the merged state back into the optimizer
               self.actor_optimizer.optimizer.load_state_dict(new_state)
               print("Transferred optimizer state for previously trainable parameters")
        
            # Print information about newly trainable parameters
            trainable_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.actor.parameters())
            print(f"After unfreezing: {trainable_params}/{total_params} parameters trainable ({trainable_params/total_params:.2%})")
        
            self.unfrozen = True
            return True
        return False            


    def perturb_layers(self, shrink_factor=0.8, epsilon=0.2):
        """Apply Shrink & Perturb selectively to LoRA/trainable parameters"""
        if not self.use_slot_att:
            return 
        with torch.no_grad():
            # Process slot attention if used
            
            for name, param in self.actor.slot_attn.named_parameters():
                if param.requires_grad or 'lora_' in name:    
                    full_name = f"slot_attn.{name}"
                    if full_name in self.initial_weights:
                        # Apply shrink
                        param.data.mul_(shrink_factor)
                        # Apply perturb using stored initialization
                        param.data.add_(epsilon * self.initial_weights[full_name].to(param.device))


        
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)


    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_cells_actor, rnn_states_critic, rnn_cells_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        param cent_obs (np.ndarray): centralized input to the critic.
         :param obs (np.ndarray): local agent inputs to the actor.
         :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
         :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
         :param masks: (np.ndarray) denotes points at which RNN states should be reset.
         :param available_actions: (np.ndarray) denotes which actions are available to agent
                                   (if None, all actions available)
         :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

         :return values: (torch.Tensor) value function predictions.
         :return actions: (torch.Tensor) actions to take.
         :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
         :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
         :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        # Convert numpy arrays to PyTorch tensors and move them to the correct device
        cent_obs = torch.tensor(cent_obs).to(self.device)
        obs = torch.tensor(obs).to(self.device)
        rnn_states_actor = torch.tensor(rnn_states_actor).to(self.device)
        rnn_cells_actor = torch.tensor(rnn_cells_actor).to(self.device)
        rnn_states_critic = torch.tensor(rnn_states_critic).to(self.device)
        rnn_cells_critic = torch.tensor(rnn_cells_critic).to(self.device)
        masks = torch.tensor(masks).to(self.device)

        if available_actions is not None:
            available_actions = torch.tensor(available_actions).to(self.device)

        # Now call the actor and critic with tensors on the correct device
        actions, action_log_probs, rnn_states_actor, rnn_cells_actor = self.actor(obs,
                                                                                  rnn_states_actor,
                                                                                  rnn_cells_actor,
                                                                                  masks,
                                                                                  available_actions,
                                                                                  deterministic
                                                                                  )

        values, rnn_states_critic, rnn_cells_critic = self.critic(cent_obs, rnn_states_critic, rnn_cells_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_cells_actor, rnn_states_critic, rnn_cells_critic

    def get_values(self, cent_obs, rnn_states_critic, rnn_cells_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _, _ = self.critic(cent_obs, rnn_states_critic, rnn_cells_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_cells_actor, rnn_states_critic, rnn_cells_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     rnn_cells_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _, _ = self.critic(cent_obs, rnn_states_critic, rnn_cells_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, rnn_cells_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor, rnn_cells_actor = self.actor(obs, rnn_states_actor, rnn_cells_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor, rnn_cells_actor
