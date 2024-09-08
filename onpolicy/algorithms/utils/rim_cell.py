import torch
import torch.nn as nn
import math
from modularity import Identity
import numpy as np
import torch.multiprocessing as mp

from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.lstm import LSTMLayer
from util import weight_init
from positional_encoder import SinusoidalPosition


class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class GroupLinearLayer(nn.Module):

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()
        self.limit = 1.0 / math.sqrt(din)

        self.w = nn.Parameter(torch.empty(num_blocks, din, dout))
        with torch.no_grad():
            torch.nn.init.uniform_(self.w, -self.limit, self.limit)

    def forward(self, x):
        x = x.permute(1, 0, 2)

        x = torch.bmm(x, self.w)
        return x.permute(1, 0, 2)


class GroupLSTMCell(nn.Module):
    """
    GroupLSTMCell can compute the operation of N LSTM Cells at once.
    """

    def __init__(self, inp_size, hidden_size, num_lstms):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size

        self.i2h = GroupLinearLayer(inp_size, 4 * hidden_size, num_lstms)
        self.h2h = GroupLinearLayer(hidden_size, 4 * hidden_size, num_lstms)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hid_state):
        """
        input: x (batch_size, num_lstms, input_size)
               hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
        output: h (batch_size, num_lstms, hidden_state)
                c ((batch_size, num_lstms, hidden_state))
        """
        h, c = hid_state
        preact = self.i2h(x) + self.h2h(h)

        gates = preact[:, :, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, :, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :, :self.hidden_size]
        f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, :, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t


class GroupGRUCell(nn.Module):
    """
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """

    def __init__(self, input_size, hidden_size, num_grus):
        super(GroupGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = GroupLinearLayer(input_size, 3 * hidden_size, num_grus)
        self.h2h = GroupLinearLayer(hidden_size, 3 * hidden_size, num_grus)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data = torch.ones(w.data.size())  # .uniform_(-std, std)

    def forward(self, x, hidden):
        """
		input: x (batch_size, num_grus, input_size)
			   hidden (batch_size, num_grus, hidden_size)
		output: hidden (batch_size, num_grus, hidden_size)
        """
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class RIMCell(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_units, k, args):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_units = num_units
        self.rnn_cell = args.rnn_attention_module
        self.input_key_size = 64
        self.key_size = self.input_key_size
        self.k = k
        self.num_input_heads = 1
        self.num_comm_heads = 4

        self.use_slot_att = args.use_slot_att
        self.input_query_size = 64
        self.input_value_size = self.hidden_size if self.use_slot_att else 400

        self.input_dropout_rate = args.drop_out
        self.comm_dropout_rate = args.drop_out

        self.comm_key_size = 32
        self.comm_query_size = 32
        self.comm_value_size = 100

        self.key = nn.Linear(input_size, self.num_input_heads * self.input_query_size).to(self.device)
        self.value = nn.Linear(input_size, self.num_input_heads * self.input_value_size).to(self.device)

        self.use_input_att = args.use_input_att
        self.use_com_att = args.use_com_att
        self.use_x_reshape = args.use_x_reshape
        if self.rnn_cell == 'LSTM':

            self.rnn = nn.ModuleList(
                [LSTMLayer(self.input_value_size, hidden_size, 1, False) for _ in range(num_units)])
        else:
            self.rnn = nn.ModuleList([RNNLayer(self.input_value_size, hidden_size, 1, False) for _ in range(num_units)])
        self.query = GroupLinearLayer(hidden_size, self.input_key_size * self.num_input_heads, self.num_units)

        self.query_ = GroupLinearLayer(hidden_size, self.comm_query_size * self.num_comm_heads, self.num_units)
        self.key_ = GroupLinearLayer(hidden_size, self.comm_key_size * self.num_comm_heads, self.num_units)
        self.value_ = GroupLinearLayer(hidden_size, self.comm_value_size * self.num_comm_heads, self.num_units)
        self.comm_attention_output = GroupLinearLayer(self.num_comm_heads * self.comm_value_size, self.hidden_size,
                                                      self.num_units)
        self.comm_dropout = nn.Dropout(p=self.comm_dropout_rate)
        self.input_dropout = nn.Dropout(p=self.input_dropout_rate)

        if self.use_x_reshape:
            self.input_linear = nn.Linear(self.hidden_size, self.input_value_size)
        else:
            self.input_linear = nn.Linear(self.num_units * self.hidden_size, self.input_value_size)

        # self.input_layer_norm  = nn.LayerNorm(self.num_units *self.hidden_size)
        self.output_layer_norm = nn.LayerNorm(self.hidden_size)
        self.to(device)
        self.apply(weight_init)

    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def input_attention_mask(self, x, h):
        """
        Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
                h (batch_size, num_units, hidden_size)
        Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
                mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
        """
        # x = self.input_layer_norm(x)
        key_layer = self.key(x)
        value_layer = self.value(x)
        query_layer = self.query(h)

        key_layer = self.transpose_for_scores(key_layer, self.num_input_heads, self.input_key_size)
        value_layer = torch.mean(self.transpose_for_scores(value_layer, self.num_input_heads, self.input_value_size),
                                 dim=1)
        query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size)
        attention_scores = torch.mean(attention_scores, dim=1)
        mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)

        not_null_scores = attention_scores[:, :, 0]
        topk1 = torch.topk(not_null_scores, self.k, dim=1)
        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, self.k)

        mask_[row_index, topk1.indices.view(-1)] = 1

        attention_probs = self.input_dropout(nn.Softmax(dim=-1)(attention_scores))
        inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)
        inputs = torch.split(inputs, 1, 1)
        return inputs, mask_

    def communication_attention(self, h, mask):
        """
        Input : h (batch_size, num_units, hidden_size)
                mask obtained from the input_attention_mask() function
        Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
        """
        query_layer = []
        key_layer = []
        value_layer = []

        query_layer = self.query_(h)
        key_layer = self.key_(h)
        value_layer = self.value_(h)

        query_layer = self.transpose_for_scores(query_layer, self.num_comm_heads, self.comm_query_size)
        key_layer = self.transpose_for_scores(key_layer, self.num_comm_heads, self.comm_key_size)
        value_layer = self.transpose_for_scores(value_layer, self.num_comm_heads, self.comm_value_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.comm_key_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        mask = [mask for _ in range(attention_probs.size(1))]
        mask = torch.stack(mask, dim=1)

        attention_probs = attention_probs * mask.unsqueeze(3)
        attention_probs = self.comm_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.comm_attention_output(context_layer)
        context_layer = self.output_layer_norm(context_layer + h)

        return context_layer

    def forward(self, x, hs, cs=None, h_masks=None):
        """
        Input : x (batch_size, 1 , input_size)
                hs (batch_size, num_units, hidden_size)
                cs (batch_size, num_units, hidden_size)
        Output: new hs, cs for LSTM
                new hs for GRU
        """
        batch_size, ep, input_size = x.shape

        if self.use_input_att:
            # Compute input attention
            null_input = torch.zeros(batch_size, 1, input_size).float().to(self.device)
            x = torch.cat((x, null_input), dim=1)
            inputs, mask = self.input_attention_mask(x, hs)
            mask = mask.unsqueeze(-1)
        else:
            if self.use_x_reshape:
                inputs = x.reshape(batch_size, ep, self.num_units, self.hidden_size).permute(2, 0, 1, 3)
            else:
                inputs = x.unsqueeze(0).repeat(self.num_units, 1, 1, 1)
            if not self.use_slot_att:
                inputs = self.input_linear(inputs)
            mask = torch.ones(batch_size, self.num_units, 1).to(self.device)

        h_old = (hs * 1.0)
        if cs is not None:
            c_old = cs * 1.0
        hs = list(torch.split(hs, 1, 1))
        if cs is not None:
            cs = list(torch.split(cs, 1, 1))

        # Compute RNN(LSTM or GRU) output
        x_out = []
        for i in range(self.num_units):
            if cs is None:
                y_t, hs[i] = self.rnn[i](inputs[i].squeeze(1), hs[i], h_masks.reshape(-1, 1))

            else:
                y_t, (hs[i], cs[i]) = self.rnn[i](inputs[i].squeeze(1), hs[i], cs[i],
                                                  h_masks.reshape(-1, 1))
            x_out.append(y_t)
        hs = torch.cat(hs, dim=1)
        x_final = torch.cat(x_out, dim=1).unsqueeze(1)
        if cs is not None:
            cs = torch.cat(cs, dim=1)

        # Block gradient through inactive units
        h_new = blocked_grad.apply(hs, mask)

        # Compute communication attention
        if self.use_com_att:
            h_new = self.communication_attention(h_new, mask.squeeze(2))

        # h_new = Identity.apply(h_new)
        hs = (mask * h_new + (1 - mask) * h_old)
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old
            return x_final, hs, cs

        # output res y (batch, 1, num_unit * hidden_size), next state hs (batch, num_unit, hidden_size)
        return x_final, hs, None


class RIM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_units, k, args):
        super().__init__()
        self.device = device
        self.n_layers = 1
        self.rnn_cell = args.rnn_attention_module
        self.num_units = num_units
        self.hidden_size = hidden_size

        self.use_pos_encoding = args.use_pos_encoding
        self.pos_encoder = SinusoidalPosition(input_size, device)

        self.x_layer_norm = nn.LayerNorm(self.hidden_size * self.num_units)

        self.rimcell = nn.ModuleList([RIMCell(self.device, input_size, hidden_size, num_units, k,
                                              args).to(self.device) if i == 0 else
                                      RIMCell(self.device, hidden_size * self.num_units, hidden_size, num_units,
                                              k, args).to(self.device) for i in
                                      range(self.n_layers)])

    def layer(self, rim_layer, x, h, c=None, direction=0, mask=None):
        batch_size = x.size(1)
        xs = list(torch.split(x, 1, dim=0))
        masks = list(torch.split(mask, 1, dim=0))

        if direction == 1:
            xs.reverse()
        hs = h.squeeze(0).view(batch_size, self.num_units, -1)
        cs = None
        if c is not None:
            cs = c.squeeze(0).view(batch_size, self.num_units, -1)
        outputs = []
        for x, mask_x in zip(xs, masks):
            x = x.squeeze(0)
            x_real, hs, cs = rim_layer(x.unsqueeze(1), hs, cs, mask_x)
            outputs.append(x_real)
        if direction == 1:
            outputs.reverse()
        outputs = torch.cat(outputs, dim=1)
        if c is not None:
            return outputs, hs.reshape(batch_size, -1), cs.reshape(batch_size, -1)
        else:
            return outputs, hs.reshape(batch_size, -1)

    def forward(self, x, h=None, c=None, masks=None):
        """
        Input: x (seq_len, batch_size, feature_size
               h (num_layers * num_directions, batch_size, hidden_size * num_units)
               c (num_layers * num_directions, batch_size, hidden_size * num_units)
        Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
                h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
        """
        batch_num = h.size(0)
        if x.size(0) == h.size(0):
            ep_len = 1
            x = x.unsqueeze(0)
        else:
            # x is a (episode_len, batch_num, -1) tensor that has been flatten to (episode_len * batch_num, -1)
            ep_len = int(x.size(0) / batch_num)
            x = x.view(ep_len, batch_num, x.size(1))

        if self.use_pos_encoding:
            pos_encoding = self.pos_encoder(ep_len).unsqueeze(1)
            x = x + pos_encoding

        # Same deal with masks
        masks = masks.view(ep_len, batch_num)

        h = h.transpose(0, 1)
        hs = torch.split(h, 1, 0) if h is not None else torch.split(
            torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(
                self.device), 1, 0)
        hs = list(hs)
        if self.rnn_cell == 'LSTM':
            c = c.transpose(0, 1)
            cs = torch.split(c, 1, 0) if c is not None else torch.split(
                torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(
                    self.device), 1, 0)
            cs = list(cs)
        else:
            cs = None

        for idx in range(self.n_layers):
            if cs is not None:
                x, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c=cs[idx], mask=masks)
            else:
                x, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c=None, mask=masks)

        x = x.transpose(0, 1).reshape(ep_len * batch_num, self.num_units * self.hidden_size)
        hs = torch.stack(hs, dim=0)
        if cs is not None:
            cs = torch.stack(cs, dim=0)
            return x, hs, cs

        return x, hs
