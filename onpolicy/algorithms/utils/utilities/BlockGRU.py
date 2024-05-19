'''
Goal1: a GRU where the weight matrices have a block structure so that information flow is constrained

Data is assumed to come in [block1, block2, ..., block_n].

Goal2: Dynamic parameter sharing between blocks (RIMs)

'''

import torch
import torch.nn as nn
from .GroupLinearLayer import GroupLinearLayer
from .sparse_attn import Sparse_attention
from .LayerNormGRUCell import LayerNormGRUCell
from onpolicy.algorithms.utils.rnn import RNNLayer

'''
Given an N x N matrix, and a grouping of size, set all elements off the block diagonal to 0.0
'''


def zero_matrix_elements(matrix, k):
    assert matrix.shape[0] % k == 0
    assert matrix.shape[1] % k == 0
    g1 = matrix.shape[0] // k
    g2 = matrix.shape[1] // k
    new_mat = torch.zeros_like(matrix)
    for b in range(0, k):
        new_mat[b * g1: (b + 1) * g1, b * g2: (b + 1) * g2] += matrix[b * g1: (b + 1) * g1, b * g2: (b + 1) * g2]

    matrix = matrix * 0.0
    matrix = matrix + new_mat


class BlockGRU(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ninp, nhid, k):
        super(BlockGRU, self).__init__()

        assert ninp % k == 0, f"ninp ({ninp}) should be divisible by k ({k})"
        assert nhid % k == 0, f"nhid ({nhid}) should be divisible by k ({k})"

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.k = k

        self.gru = RNNLayer(ninp, nhid, 1, False).to(self.device)

        self.nhid = nhid
        self.ninp = ninp
        self.to(self.device)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for the GRU cell
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def blockify_params(self):
        pl = self.gru.parameters()

        for p in pl:
            p = p.data
            if p.shape == torch.Size([self.nhid * 3]):
                pass
                '''biases, don't need to change anything here'''
            if p.shape == torch.Size([self.nhid * 3, self.nhid]) or p.shape == torch.Size([self.nhid * 3, self.ninp]):
                for e in range(0, 4):
                    zero_matrix_elements(p[self.nhid * e: self.nhid * (e + 1)], k=self.k)

    def forward(self, input, h, masks):
        hnext = self.gru(input, h, masks)
        return hnext, None


class SharedBlockGRU(nn.Module):
    """Dynamic sharing of parameters between blocks(RIM's)"""

    def __init__(self, ninp, nhid, k, n_templates):
        super(SharedBlockGRU, self).__init__()

        assert ninp % k == 0, f"ninp ({ninp}) should be divisible by k ({k})"
        assert nhid % k == 0, f"nhid ({nhid}) should be divisible by k ({k})"

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.k = k
        self.m = nhid // self.k

        self.n_templates = n_templates

        self.templates = nn.ModuleList(
                [RNNLayer(ninp, self.m, 1, False).to(self.device) for _ in range(0, self.n_templates)])

        self.nhid = nhid

        self.ninp = ninp

        self.gll_write = GroupLinearLayer(self.m, 16, self.n_templates, device=self.device)
        self.gll_read = GroupLinearLayer(self.m, 16, 1, device=self.device)
        self.sa = Sparse_attention(1).to(self.device)
        self.initialize_weights()
        self.to(self.device)

    def initialize_weights(self):
        # Initialize weights for each GRUCell template
        for template in self.templates:
            for name, param in template.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def blockify_params(self):

        return

    def forward(self, x, h, masks):

        # self.blockify_params()
        bs = h.shape[0]
        h = h.reshape(bs * self.k, self.m)

        x = x.unsqueeze(1).repeat(1, self.k, 1)
        x = x.reshape(x.shape[0] * self.k, x.shape[2])

        h = h.unsqueeze(1)
        h_read = self.gll_read(h)

        hnext_stack = []

        h_masks = masks.repeat(self.k, 1)
        for template in self.templates:
            x_out, hnext_l = template(x, h, h_masks)
            hnext_stack.append(hnext_l)

        hnext = torch.cat(hnext_stack, 1)

        write_key = self.gll_write(hnext)

        '''
        sm = nn.Softmax(2)
        att = sm(torch.bmm(h_read, write_key.permute(0, 2, 1))).squeeze(1)
        att = self.sa(att).unsqueeze(1)
        '''

        att = torch.nn.functional.gumbel_softmax(torch.bmm(h_read, write_key.permute(0, 2, 1)), tau=0.5, hard=True)
        # att = att*0.0 + 0.25

        hnext = torch.bmm(att, hnext)

        hnext = hnext.mean(dim=1)
        hnext = hnext.reshape((bs, self.k, self.m)).reshape((bs, self.k * self.m))

        return (None, hnext), att.data.reshape(bs, self.k, self.n_templates)
