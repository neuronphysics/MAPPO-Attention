import torch
import torch.nn as nn
import torch.functional as F
from onpolicy.algorithms.utils.util import init, check, calculate_conv_params
from onpolicy.algorithms.utils.cnn import CNNBase, Encoder
from onpolicy.algorithms.utils.modularity import RIM, SCOFF
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from absl import logging
from onpolicy.algorithms.utils.cnn import Decoder
import numpy as np

# ------------------------- Dane addition


LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


# -------------------------


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 obs_space,
                 action_space,
                 log_std_min,
                 log_std_max,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_version_scoff = args.use_version_scoff
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        ##Zahra added
        self.use_attention = False
        # self.use_attention = args.use_attention
        self._attention_module = args.attention_module
        print(f"value of use attention is {self.use_attention} ")

        self._obs_shape = obs_shape
        print(f"actor network observation shape {obs_shape} {len(self._obs_shape)}")

        if self.use_attention and len(self._obs_shape) >= 3:
            print(f"value of use attention is {self.use_attention} ")
            logging.info('Using attention module %s: input width: %d', self._attention_module, obs_shape[1])
            # print(f"we are using both CNN and attention module.... {obs_shape} {len(self._obs_shape)}")
            if obs_shape[0] == 3:
                input_channel = obs_shape[0]
                input_width = obs_shape[1]
                input_height = obs_shape[2]
            elif obs_shape[2] == 3:
                input_channel = obs_shape[2]
                input_width = obs_shape[0]
                input_height = obs_shape[1]
                # making parametrs of encoder for CNN compatible with different image sizes
            print(
                f"input channel and input image width in actor network c: {input_channel}, w: {input_width}, h: {input_height}")
            kernel, stride, padding = calculate_conv_params((input_width, input_height, input_channel))

            self.base = Encoder(input_channel,
                                input_height,
                                input_width,
                                self.hidden_size,
                                device,
                                max_filters=256,
                                num_layers=3, kernel_size=kernel, stride_size=stride, padding_size=padding)
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max

            # ------------------- Dane Addition
            self.decode = Decoder(input_channel,
                                  input_height,
                                  input_width,
                                  self.hidden_size,
                                  device,
                                  max_filters=256,
                                  num_layers=3, kernel_size=kernel, stride_size=stride, padding_size=padding)
            # ---------------------------------

            if self._attention_module == "RIM":
                print("We are using RIM...")
                self.rnn = RIM(device, self.hidden_size, self.hidden_size, 6, 4, rnn_cell='GRU', n_layers=1,
                               bidirectional=False, batch_first=True)
            elif self._attention_module == "SCOFF":
                print("We are using SCOFF...")
                self.rnn = SCOFF(device, self.hidden_size, self.hidden_size, 4, 3, num_templates=2, rnn_cell='GRU',
                                 n_layers=1, bidirectional=False, batch_first=False, version=self._use_version_scoff)

        elif not self.use_attention:
            print(f"value of use attention is {self.use_attention} ")
            base = CNNBase if len(obs_shape) >= 3 else MLPBase
            logging.info("observation space %s number of dimensions of observation space is %d", str(obs_space.shape),
                         len(obs_shape))
            if len(obs_shape) == 3:
                logging.info('Not using any attention module, input width: %d ', obs_shape[1])
            self.base = base(args, obs_shape)

            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                print("We are using LSTM...")
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, 2 * action_space[0])
        )  # unknown why trunk is necessary - Dane

        # weights are shared? - Dane
        self.outputs = dict()

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, compute_pi=True, compute_log_pi=True, available_actions=None,
                deterministic=False):
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
        obs = check(obs).to(**self.tpdv)  # encoder was detached in SAC-AE

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.use_attention and len(self._obs_shape) >= 3:
            actor_features = self.base(obs)
            recon_features = self.decode(actor_features)

            print(
                f"actor features shape.... {actor_features.shape} {rnn_states.shape}")  # torch.Size([9, 64]) torch.Size([9, 1, 64])
            actor_features, rnn_states = self.rnn(actor_features, rnn_states)
            print(
                f"actor features shape after normal RNN in an actor network (attention).... {actor_features[0].shape} {rnn_states[0].shape}")

        else:

            actor_features = self.base(obs)
            recon_features = self.decode(actor_features)
            print(f"actor features shape base CNN and RNN.... {actor_features.shape} {rnn_states.shape}")
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
                print(
                    f"actor features shape after normal RNN in an actor network (lstm).... {actor_features.shape} {rnn_states.shape}")
                rnn_states = rnn_states.permute(1, 0, 2)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        # ----------------------- Dane Addition

        mu, log_std = self.trunk(actor_features).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        # -----------------------
        # actions, action_log_probs =  mu, pi

        return actions, action_log_probs, rnn_states, log_pi, log_std, recon_features

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
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
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        recon_features = self.decode(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.act.evaluate_actions_trpo(
                actor_features,
                action, available_actions,
                active_masks=
                active_masks if self._use_policy_active_masks
                else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy, recon_features = self.act.evaluate_actions(actor_features,
                                                                       action, available_actions,
                                                                       active_masks=
                                                                       active_masks if self._use_policy_active_masks
                                                                       else None)

        return action_log_probs, dist_entropy, recon_features


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
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        ## Zahra added
        self._use_version_scoff = args.use_version_scoff
        self.use_attention = False
        # self.use_attention = args.use_attention
        self._attention_module = args.attention_module

        self._obs_shape = cent_obs_shape
        print(f"critic network observation shape {cent_obs_shape} {len(self._obs_shape)}")
        if self.use_attention and len(self._obs_shape) >= 3:

            print(f"value of use attention in critic network is {self.use_attention} ")
            if self._obs_shape[0] == 3:
                input_channel = cent_obs_shape[0]
                input_width = cent_obs_shape[1]
                input_height = cent_obs_shape[2]
            elif self._obs_shape[2] == 3:
                input_channel = cent_obs_shape[2]
                input_width = cent_obs_shape[0]
                input_height = cent_obs_shape[1]
                # making parametrs of encoder for CNN compatible with different image sizes
            print(
                f"input channel and input image width in critic network c:{input_channel}, w: {input_width}, h: {input_height} observation: {cent_obs_shape}")
            kernel, stride, padding = calculate_conv_params((input_width, input_height, input_channel))

            self.base = Encoder(input_channel, input_height, input_width, self.hidden_size, device, max_filters=256,
                                num_layers=3, kernel_size=kernel, stride_size=stride, padding_size=padding)
            if self._attention_module == "RIM":

                self.rnn = RIM(device, self.hidden_size, self.hidden_size, 6, 4, rnn_cell='GRU', n_layers=1,
                               bidirectional=False, batch_first=True)

            elif self._attention_module == "SCOFF":
                print(
                    f"we are using SCOFF attention module in critic network.... {cent_obs_shape} {len(self._obs_shape)}")
                self.rnn = SCOFF(device, self.hidden_size, self.hidden_size, 4, 3, num_templates=2, rnn_cell='GRU',
                                 n_layers=1, bidirectional=False, batch_first=False, version=self._use_version_scoff)
        elif not self.use_attention:
            print(f"value of use attention in critic network is {self.use_attention} ")
            base = CNNBase if len(cent_obs_shape) >= 3 else MLPBase
            self.base = base(args, cent_obs_shape)

            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                print("We are using LSTM...")
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
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
        masks = check(masks).to(**self.tpdv)
        if self.use_attention and len(self._obs_shape) == 3:
            print(f"critic features shape of shared observation before cnn.... {cent_obs.shape} {masks.shape}")
            critic_features = self.base(cent_obs)
            print(f"critic features shape before rnn.... {critic_features.shape} {rnn_states.shape}")
            critic_features, rnn_states = self.rnn(critic_features, rnn_states)
            print(
                f"critic features shape after rnn using attention.... {critic_features.shape} {rnn_states[0].shape}")  # torch.Size([1, rollout,hidden_size]) torch.Size([1, rollout, hidden_size])
        else:

            critic_features = self.base(cent_obs)
            print(f"critic features shape before rnn using (normal rnn).... {critic_features.shape} {rnn_states.shape}")
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
                print(
                    f"critic features shape after rnn using (normal rnn).... {critic_features.shape} {rnn_states.shape}")  # torch.Size([rollout,hidden_size]) torch.Size([rollout, 1, hidden_size])
                critic_features = critic_features.unsqueeze(0)
                rnn_states = rnn_states.permute(1, 0, 2)
        values = self.v_out(critic_features)

        return values, rnn_states