import torch
from torch import nn
import math
from einops import rearrange


class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""

    def __init__(self, dim, device, min_timescale=2., max_timescale=1e4):
        super().__init__()
        self.device = device

        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.).to(self.device)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb
