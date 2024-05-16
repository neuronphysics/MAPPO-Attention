import time
import torch
from torch import nn
from onpolicy.algorithms.utils.SCOFF.blocks_core_scoff import BlocksCore


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False, use_cudnn_version=False, use_adaptive_softmax=False, cutoffs=None,
                 discrete_input=False, n_templates=0, share_inp=True, share_comm=True,
                 memory_mlp=4, num_blocks=6, update_topk=4, memorytopk=4,
                 use_gru=False, do_rel=False, num_modules_read_input=4, inp_heads=4,
                 device=None, memory_slots=4, memory_head_size=16,
                 num_memory_heads=4, attention_out=340, version=1, step_att=True):

        super(RNNModel, self).__init__()
        self.device = device
        self.topk = update_topk
        self.memorytopk = memorytopk
        self.num_modules_read_input = num_modules_read_input
        self.inp_heads = inp_heads
        self.use_cudnn_version = use_cudnn_version
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
            "memory_slots": memory_slots,
            "memory_head_size": memory_head_size,
            "num_memory_heads": num_memory_heads,
            "share_inp": share_inp,
            "share_comm": share_comm,
            "memory_mlp": memory_mlp,
            "attention_out": attention_out,
            "version": version,
            "step_att": step_att,
            "topk": update_topk,
            "memorytopk": self.memorytopk,
            "num_blocks": num_blocks,
            "n_templates": n_templates,
            "num_modules_read_input": num_modules_read_input,
            "inp_heads": inp_heads,
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
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.rule_emb.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_blocks_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.rule_emb.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
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
        step_att = self.args_to_init_blocks["step_att"]
        topk = self.args_to_init_blocks["topk"]
        memorytopk = self.args_to_init_blocks["memorytopk"]
        num_blocks = self.args_to_init_blocks["num_blocks"]
        n_templates = self.args_to_init_blocks["n_templates"]
        num_modules_read_input = self.args_to_init_blocks["num_modules_read_input"]
        inp_heads = self.args_to_init_blocks["inp_heads"]
        nhid = self.args_to_init_blocks["nhid"]

        if self.discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)

        bc_lst = []

        bc_lst.append(
            BlocksCore(nhid, 1, num_blocks, topk, memorytopk, step_att, num_modules_read_input, inp_heads,
                       do_gru=use_gru,
                       do_rel=do_rel, device=device, n_templates=n_templates,
                       share_inp=share_inp, share_comm=share_comm, memory_mlp=memory_mlp,
                       memory_slots=memory_slots, num_memory_heads=num_memory_heads,
                       memory_head_size=memory_head_size, attention_out=attention_out,
                       version=version))
        self.bc_lst = nn.ModuleList(bc_lst)

        self.decoder = nn.Linear(self.nhid, ntoken)
        if tie_weights:
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

    def forward(self, input, hidden, masks=None):
        extra_loss = 0.0
        timesteps, batch_size, _ = input.shape

        emb = self.encoder(input)

        if True:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = []
            for idx_layer in range(0, self.nlayers):
                output = []
                bmasklst = []
                template_attn = []

                self.bc_lst[idx_layer].blockify_params()

                hx = hidden[0]

                if self.do_rel:
                    self.bc_lst[idx_layer].reset_relational_memory(input.shape[1])

                for idx_step in range(input.shape[0]):
                    hx, mask, bmask, temp_attn = self.bc_lst[idx_layer](layer_input[idx_step], hx,
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

            new_hidden = torch.stack(new_hidden)

        block_mask = bmask.squeeze(0)

        output = self.drop(output)
        dec = output.view(output.size(0) * output.size(1), self.nhid)
        dec = self.decoder(dec)
        return dec.view(output.size(0), output.size(1), dec.size(1)), new_hidden, extra_loss, block_mask, template_attn

    def init_hidden(self, bsz):
        weight = next(self.bc_lst[0].block_lstm.parameters())
        if True or self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
            # return (weight.new(self.nlayers, bsz, self.nhid).normal_(),
            #        weight.new(self.nlayers, bsz, self.nhid).normal_())
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
