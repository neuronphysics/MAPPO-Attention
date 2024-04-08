import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNCellBase
import torch.nn.functional as F
import math

class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.hidden_size = hidden_size
        self.a2 = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z, 1)
        sigma = torch.std(z, dim=1, unbiased=False)

        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a2.expand_as(ln_out) + self.b2.expand_as(ln_out)

        return ln_out

class LayerNormGRUCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))

        self.reset_ln = nn.LayerNorm(self.hidden_size)
        self.input_ln = nn.LayerNorm(self.hidden_size)
        self.new_gate_ln = nn.LayerNorm(self.hidden_size)

        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):

        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden, self.weight_hh, self.bias_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(self.reset_ln(i_r + h_r))
        inputgate = torch.sigmoid(self.input_ln(i_i + h_i))
        newgate = torch.tanh(self.new_gate_ln(i_n + resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)

        return hy