import torch.nn as nn
import torch
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from SCOFF.rnn_models_scoff import RNNModel as RNNModelScoff


class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input * 1.0

    def backward(ctx, grad_output):
        return grad_output * 1.0


class SCOFF(nn.Module):
    def __init__(self,
                 device,
                 input_size,
                 hidden_size,
                 num_units,
                 k,
                 memory_topk,
                 num_modules_read_input,
                 inp_heads,
                 share_comm,
                 share_inp,
                 memory_mlp,
                 memory_slots,
                 memory_head_size,
                 num_memory_heads,
                 num_templates=0,
                 rnn_cell='GRU',
                 n_layers=1,
                 version=0,
                 attention_out=85,
                 bidirectional=False,
                 batch_first=False,
                 dropout=0.0,
                 step_att=True,
                 do_relational_memory=False,
                 ):
        super().__init__()
        """
        - Wrappper for SCOFF.
        - Mirrors nn.LSTM or nn.GRU
        - supports bidirection and multiple layers
        - Option to specify num_rules and rule_time_steps.

        Parameters:
            device: 'cuda' or 'cpu'
            input_size
            hidden_size
            num_units: Number of RIMs
            k: topk
            rnn_cell: 'LSTM' or 'GRU' (default = LSTM)
            n_layers: num layers (default = 1)
            bidirectional: True or False (default = False)
            num_rules: number of rules (default = 0)
            rule_time_steps: Number of times to apply rules per time step (default = 0)
        """
        if input_size % num_units != 0:
            print('ERROR: input_size should be evenly divisible by num_units')
            exit()
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1
        self.rnn_cell = rnn_cell
        self.num_units = num_units
        self.hidden_size = hidden_size // num_units
        self.batch_first = batch_first
        self.do_rel = do_relational_memory

        self.scoff_cell = RNNModelScoff(rnn_cell, input_size, hidden_size, hidden_size, nlayers=1,
                                        n_templates=num_templates, tie_weights=False, num_blocks=num_units,
                                        update_topk=k,
                                        use_cudnn_version=False, use_adaptive_softmax=False, discrete_input=False,
                                        use_gru=rnn_cell == 'GRU', version=version, memorytopk=memory_topk,
                                        num_modules_read_input=num_modules_read_input, inp_heads=inp_heads,
                                        attention_out=attention_out, dropout=dropout, share_comm=share_comm,
                                        share_inp=share_inp,
                                        memory_mlp=memory_mlp, memory_slots=memory_slots,
                                        memory_head_size=memory_head_size,
                                        num_memory_heads=num_memory_heads,
                                        step_att=step_att,
                                        do_rel=self.do_rel).to(self.device)

    def layer(self, rim_layer, x, h, direction=0, message_to_rule_network=None, masks=None):
        batch_size = x.size(1)
        xs = list(torch.split(x, 1, dim=0))

        if direction == 1: xs.reverse()
        xs = torch.cat(xs, dim=0)

        hidden = h
        entropy = 0

        outputs, hidden, _, _, _, entropy_ = rim_layer(xs, hidden,
                                                       message_to_rule_network=message_to_rule_network, masks=masks)
        entropy += entropy_

        hs = hidden  # self.rim_inverse_transform_hidden(hidden)

        outputs = list(torch.split(outputs, 1, dim=0))

        if direction == 1: outputs.reverse()
        outputs = torch.cat(outputs, dim=0)

        hs_ = hs.reshape(batch_size, -1)
        return outputs, hs_, entropy

    def forward(self, x, h, masks=None):
        """
        Input: x (seq_len, batch_size, feature_size
               hidden (num_layers * num_directions, batch_size, hidden_size * num_units)
        Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
                h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
        """

        batch_num = h.size(0)
        if x.size(0) == h.size(0):
            ep_len = 1
            x = x.unsqueeze(0)
        else:
            # x is a (episode_len, batch_num, -1) tensor that has been flattened to (episode_len * batch_num, -1)
            ep_len = int(x.size(0) / batch_num)
            x = x.view(ep_len, batch_num, x.size(1))

        # Same deal with masks
        masks = masks.view(ep_len, batch_num)

        h = h.transpose(0, 1)

        x_fw, hs, _, _, _ = self.scoff_cell(x, h.unsqueeze(0), masks=masks)

        # x should be (seq, batch, input_dim)
        x_fw = x_fw.reshape(ep_len * batch_num, self.num_units * self.hidden_size)
        return x_fw, hs
