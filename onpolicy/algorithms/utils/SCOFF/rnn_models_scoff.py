import math
import time
import torch
from torch import nn
from onpolicy.algorithms.utils.SCOFF.blocks_core_scoff import BlocksCore


def build_2d_sincos_pos_enc(grid, dim):
    """Fixed 2D sine/cosine positional encoding for a (H, W) token grid -> (H*W, dim).
    Row-major order, matching CNNLayer's flatten(2).transpose(1, 2)."""
    h, w = grid
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)          # (P, 2)
    n_freq = (dim + 3) // 4                                                # sin+cos per axis
    freqs = torch.exp(torch.arange(n_freq, dtype=torch.float32)
                      * (-math.log(10000.0) / max(n_freq - 1, 1)))
    ang = coords.unsqueeze(-1) * freqs                                     # (P, 2, F)
    pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(coords.shape[0], -1)
    return pe[:, :dim].contiguous()


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, args, dropout=0.5,
                 tie_weights=False, use_cudnn_version=False, use_adaptive_softmax=False, cutoffs=None,
                 discrete_input=False, n_templates=0, num_blocks=6, update_topk=4, use_gru=False, do_rel=False,
                 device=None, attention_out=340, version=1, token_dim=None, token_grid=None,
                 token_count=None):

        super(RNNModel, self).__init__()
        self.args = args
        self.num_modules_read_input = args.scoff_num_modules_read_input
        self.inp_heads = args.scoff_inp_heads
        self.share_comm = args.scoff_share_comm
        self.share_inp = args.scoff_share_inp
        self.memory_mlp = args.scoff_memory_mlp
        self.memory_slots = args.scoff_memory_slots
        self.memory_head_size = args.scoff_memory_head_size
        self.num_memory_heads = args.scoff_num_memory_heads

        self.device = device
        self.topk = update_topk
        self.memorytopk = args.scoff_memory_topk
        self.use_cudnn_version = use_cudnn_version
        self.version = version
        self.token_dim = token_dim
        self.token_grid = token_grid
        if token_count is None and token_grid is not None:
            token_count = token_grid[0] * token_grid[1]
        self.token_count = token_count
        self.drop = nn.Dropout(dropout)
        self.n_templates = n_templates
        self.do_rel = do_rel

        self.args_to_init_blocks = {
            "ntoken": ntoken,
            "ninp": ninp,
            "use_gru": use_gru,
            "tie_weights": tie_weights,
            "do_rel": do_rel,
            "device": device,
            "memory_slots": self.memory_slots,
            "memory_head_size": self.memory_head_size,
            "num_memory_heads": self.num_memory_heads,
            "share_inp": self.share_inp,
            "share_comm": self.share_comm,
            "memory_mlp":self. memory_mlp,
            "attention_out": attention_out,
            "version": version,
            "topk": update_topk,
            "memorytopk": self.memorytopk,
            "num_blocks": num_blocks,
            "n_templates": n_templates,
            "num_modules_read_input": self.num_modules_read_input,
            "inp_heads": self.inp_heads,
            "nhid": nhid,
        }
        self.num_blocks = num_blocks
        self.nhid = nhid
        self.block_size = nhid // self.num_blocks
        self.discrete_input = discrete_input
        self.use_adaptive_softmax = use_adaptive_softmax
        self.bc_lst = None
        self.sigmoid = nn.Sigmoid()
        self.decoder = None

        self.rnn_type = rnn_type

        self.nlayers = nlayers

        self.prior_lst = None
        self.inf_ = None
        self.prior_ = None

        self.init_blocks()

    def init_weights(self):
        initrange = 0.1
        if self.encoder is not None:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax and self.decoder is not None:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_blocks_weights(self):
        initrange = 0.1
        if self.encoder is not None:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax and self.decoder is not None:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_blocks(self):
        ntoken = self.args_to_init_blocks["ntoken"]
        ninp = self.args_to_init_blocks["ninp"]
        use_gru = self.args_to_init_blocks["use_gru"]
        tie_weights = self.args_to_init_blocks["tie_weights"]
        device = self.args_to_init_blocks["device"]
        do_rel = self.args_to_init_blocks["do_rel"]
        memory_slots = self.args_to_init_blocks["memory_slots"]
        memory_head_size = self.args_to_init_blocks["memory_head_size"]
        num_memory_heads = self.args_to_init_blocks["num_memory_heads"]
        share_inp = self.args_to_init_blocks["share_inp"]
        share_comm = self.args_to_init_blocks["share_comm"]
        memory_mlp = self.args_to_init_blocks["memory_mlp"]
        attention_out = self.args_to_init_blocks["attention_out"]
        version = self.args_to_init_blocks["version"]
        topk = self.args_to_init_blocks["topk"]
        memorytopk = self.args_to_init_blocks["memorytopk"]
        num_blocks = self.args_to_init_blocks["num_blocks"]
        n_templates = self.args_to_init_blocks["n_templates"]
        num_modules_read_input = self.args_to_init_blocks["num_modules_read_input"]
        inp_heads = self.args_to_init_blocks["inp_heads"]
        nhid = self.args_to_init_blocks["nhid"]

        if self.version == 2:
            # Token input path: no dense encoder -- a Linear over the concatenation
            # would mix tokens and destroy entity structure before the object files
            # ever compete for it. Location information comes from a fixed 2D
            # sinusoidal positional encoding instead.
            self.encoder = None
            if self.token_dim is None:
                raise ValueError("version 2 requires token_dim")
            if self.token_grid is not None:
                self.register_buffer(
                    "pos_enc", build_2d_sincos_pos_enc(self.token_grid, self.token_dim),
                    persistent=False)
            else:
                self.pos_enc = None
        elif self.discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)

        bc_lst = []

        bc_lst.append(
            BlocksCore(nhid, 1, num_blocks, topk, memorytopk, num_modules_read_input, inp_heads,
                       do_gru=use_gru,
                       do_rel=do_rel, device=device, n_templates=n_templates,
                       share_inp=share_inp, share_comm=share_comm, memory_mlp=memory_mlp,
                       memory_slots=memory_slots, num_memory_heads=num_memory_heads,
                       memory_head_size=memory_head_size, attention_out=attention_out,
                       version=version, args=self.args, token_dim=self.token_dim,
                       token_count=self.token_count))
        self.bc_lst = nn.ModuleList(bc_lst)

        if self.version == 2:
            self.decoder = None   # version 2 returns the object-file states directly
        else:
            self.decoder = nn.Linear(self.nhid, ntoken)
        if tie_weights and self.decoder is not None:
            if self.nhid != ninp:
                raise ValueError('When using the tied flag, '
                                 'nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_blocks_weights()

    def reparameterize(self, mu, logvar):
        if True:  # self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input, hidden, cs, masks=None):
        extra_loss = 0.0
        timesteps, batch_size, _ = input.shape

        if self.version == 2:
            tokens = input.view(timesteps, batch_size, -1, self.token_dim)
            if self.pos_enc is not None:
                if tokens.shape[2] != self.pos_enc.shape[0]:
                    raise ValueError(
                        f"got {tokens.shape[2]} tokens but the positional encoding was "
                        f"built for {self.pos_enc.shape[0]} grid positions")
                tokens = tokens + self.pos_enc
            emb = tokens
        else:
            emb = self.encoder(input)

        if True:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = []
            new_c = []
            for idx_layer in range(0, self.nlayers):
                output = []
                bmasklst = []
                template_attn = []

                self.bc_lst[idx_layer].blockify_params()

                hx = hidden[0]
                if cs is not None:
                    cx = cs[0]
                else:
                    cx = None

                if self.do_rel:
                    self.bc_lst[idx_layer].reset_relational_memory(input.shape[1])

                for idx_step in range(input.shape[0]):
                    hx, cx, mask, bmask, temp_attn = self.bc_lst[idx_layer](layer_input[idx_step], hx, cx,
                                                                        h_masks=masks[idx_step])
                    hx = hx.unsqueeze(0)

                    output.append(hx)
                    bmasklst.append(bmask)
                    template_attn.append(temp_attn)

                output = torch.cat(output, dim=0)
                bmask = torch.stack(bmasklst)
                if type(template_attn[0]) != type(None):
                    template_attn = torch.stack(template_attn)

                layer_input = output
                new_hidden.append(hx)
                new_c.append(cx)

            new_hidden = torch.stack(new_hidden)
            if cs is not None:
                new_c = torch.stack(new_c)
            else:
                new_c = None

        block_mask = bmask.squeeze(0)

        output = self.drop(output)
        if self.decoder is None:
            dec = output
        else:
            dec = self.decoder(output.view(output.size(0) * output.size(1), self.nhid))
            dec = dec.view(output.size(0), output.size(1), dec.size(1))
        return dec, new_hidden, new_c, extra_loss, block_mask, template_attn

    def init_hidden(self, bsz):
        weight = next(self.bc_lst[0].block_lstm.parameters())
        if True or self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
            # return (weight.new(self.nlayers, bsz, self.nhid).normal_(),
            #        weight.new(self.nlayers, bsz, self.nhid).normal_())
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)