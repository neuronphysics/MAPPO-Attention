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
                 use_gru=False, do_rel=False, num_modules_read_input=2, inp_heads=1,
                 device=None, memory_slots=4, memory_head_size=16,
                 num_memory_heads=4, attention_out=512, version=1, step_att=True,
                 num_rules=0, rule_time_steps=0, perm_inv=True, application_option=3, use_dropout=True,
                 rule_selection='gumble'):

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
        self.use_dropout = use_dropout

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
            "perm_inv": perm_inv,
        }
        self.num_blocks = num_blocks
        self.nhid = nhid
        self.block_size = nhid // self.num_blocks
        self.discrete_input = discrete_input
        self.use_adaptive_softmax = use_adaptive_softmax
        self.bc_lst = None
        self.sigmoid = nn.Sigmoid()
        self.decoder = None

        self.rule_selection = rule_selection

        self.application_option = application_option

        self.perm_inv = perm_inv

        self.num_rules = num_rules
        self.rule_time_steps = rule_time_steps

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
        perm_inv = self.args_to_init_blocks["perm_inv"]

        if self.discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)

        bc_lst = []

        bc_lst.append(
            BlocksCore(ninp, nhid, 1, num_blocks, topk, memorytopk, step_att, num_modules_read_input, inp_heads,
                       do_gru=use_gru,
                       do_rel=do_rel, perm_inv=perm_inv, device=device, n_templates=n_templates,
                       share_inp=share_inp, share_comm=share_comm, memory_mlp=memory_mlp,
                       memory_slots=memory_slots, num_memory_heads=num_memory_heads,
                       memory_head_size=memory_head_size, attention_out=attention_out,
                       version=version, num_rules=self.num_rules, rule_time_steps=self.rule_time_steps,
                       application_option=self.application_option, rule_selection=self.rule_selection))
        self.bc_lst = nn.ModuleList(bc_lst)

        if tie_weights:
            if self.nhid != ninp:
                raise ValueError('When using the tied flag, '
                                 'nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight



    def reparameterize(self, mu, logvar):
        if True:  # self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input, hidden, message_to_rule_network=None, masks=None):
        extra_loss = 0.0

        # input_to_blocks = input.reshape(batch_size, self.block_size * self.num_blocks)
        emb = self.drop(self.encoder(input))
        timesteps, batch_size, _ = emb.shape

        entropy = 0

        if True:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = []
            for idx_layer in range(0, self.nlayers):
                # print('idx layer', idx_layer)
                output = []
                bmasklst = []
                template_attn = []

                self.bc_lst[idx_layer].blockify_params()

                hx = hidden[idx_layer]

                if self.do_rel:
                    self.bc_lst[idx_layer].reset_relational_memory(input.shape[1])

                for idx_step in range(input.shape[0]):
                    hx, mask, bmask, temp_attn, entropy_ = self.bc_lst[idx_layer](layer_input[idx_step], hx,
                                                                                  h_masks=masks[idx_step],
                                                                                  message_to_rule_network=message_to_rule_network)
                    hx = hx.unsqueeze(0)

                    entropy += entropy_
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

        return output, new_hidden, extra_loss, block_mask, template_attn, entropy


def init_hidden(self, bsz):
    weight = next(self.bc_lst[0].block_lstm.parameters())
    if True or self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
        # return (weight.new(self.nlayers, bsz, self.nhid).normal_(),
        #        weight.new(self.nlayers, bsz, self.nhid).normal_())
    else:
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
