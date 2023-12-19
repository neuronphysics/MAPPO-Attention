from collections import OrderedDict
from itertools import chain

import numpy as np
import torch
from torch import nn

class CausalTransitionModelLSTM(nn.Module):
    """Main module for a Recurrent Causal transition model.
    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
        rim: If False uses LSTM else RIMs (goyal et al)
    """
    def __init__(self, embedding_dim_per_object, input_dims, hidden_dim, action_dim,
                 num_objects, input_shape=[3, 50, 50], predict_diff=True, encoder='large',
                 graph=None, modular=False, vae=False, rim = False, scoff = False,
                 multiplier=1):
        super(CausalTransitionModelLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.input_shape = input_shape
        self.modular = modular
        self.predict_diff = predict_diff
        self.vae = vae
        self.graph = graph

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        if self.modular:
            self.embedding_dim = embedding_dim_per_object
            flat = False
        else:
            self.embedding_dim = embedding_dim_per_object * num_objects
            flat = True

        if encoder == 'small':
            obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            width_height = np.array(width_height)
            width_height = width_height // 10
            self.decoder = DecoderCNNSmall(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        elif encoder == 'medium':
            obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            width_height = np.array(width_height)
            width_height = width_height // 5
            self.decoder = DecoderCNNMedium(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        elif encoder == 'large':
            obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            self.decoder = DecoderCNNLarge(
                            input_dim=self.embedding_dim,
                            num_objects=self.num_objects,
                            hidden_dim=self.hidden_dim//2,
                            output_size=self.input_shape,
                            flat_state=flat)

        if self.modular:
            obj_encoder = EncoderMLP(
                input_dim=np.prod(width_height),
                hidden_dim=hidden_dim,
                output_dim=self.embedding_dim,
                num_objects=num_objects)
        else:
            if self.vae:
                obj_encoder = EncoderMLP(
                    input_dim=np.product(width_height)*self.num_objects,
                    output_dim=self.embedding_dim * 2,
                    hidden_dim=hidden_dim, num_objects=num_objects,
                    flatten_input=True)
            else:
                obj_encoder = EncoderMLP(
                    input_dim=np.product(width_height)*self.num_objects,
                    output_dim=self.embedding_dim,
                    hidden_dim=hidden_dim, num_objects=num_objects,
                    flatten_input=True)
            
            self.action_encoder = nn.Linear(self.num_objects * self.action_dim, self.hidden_dim)
            self.obs_encoder = nn.Linear(self.embedding_dim, self.hidden_dim)
            self.lstm = False
            if rim == True:
                self.rim = True
                self.transition_nets =  RIM('cuda', 2 * self.hidden_dim, 600, 6, 4, rnn_cell = 'GRU', n_layers = 1, bidirectional = False)
                self.transition_linear = nn.Linear(600, self.embedding_dim)
            elif scoff == True:
                self.transition_nets = SCOFF('cuda', 2 * self.hidden_dim, 600, 4, 3, num_templates = 2, rnn_cell = 'GRU', n_layers = 1, bidirectional = False, version=1)
                self.transition_linear = nn.Linear(600, self.embedding_dim)
            else:
                self.lstm = True
                self.transition_nets = nn.LSTM(2 * self.hidden_dim, 600)
                self.transition_linear = nn.Linear(600, self.embedding_dim)

        self.encoder = nn.Sequential(OrderedDict(
            obj_extractor=obj_extractor,
            obj_encoder=obj_encoder))

        self.width = width_height[0]
        self.height = width_height[1]

    def transition_parameters(self):
        parameters = []
        if isinstance(self.transition_nets, list):
            for net in self.transition_nets:
                parameters = chain(parameters, net.parameters())
        else:
            return list(self.transition_nets.parameters()) + list(self.transition_linear.parameters())

        return parameters

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()

    def encode(self, obs):
        enc = self.encoder(obs)
        if self.vae:
            mu = enc[:, :self.embedding_dim]
            logvar = enc[:, self.embedding_dim:]
            if self.training:
                sigma = torch.exp(0.5 * logvar)
                eps = torch.randn_like(sigma)
                z = mu + eps * sigma
            else:
                z = mu
            return z, (mu, logvar)
        else:
            return enc, None

    def transition(self, state, action, hidden):
        encoded_action = self.action_encoder(action)
        encoded_state = self.obs_encoder(state)
        
        x = torch.cat((encoded_state, encoded_action), dim = 1)
        x = x.unsqueeze(0)
        if not self.lstm:
            x, hidden, _ = self.transition_nets(x, hidden)
        else:
            x, hidden = self.transition_nets(x, hidden)

        x = self.transition_linear(x)
        x = x.squeeze(0)

        if self.predict_diff:
            x = state + x

        return x, hidden

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))
