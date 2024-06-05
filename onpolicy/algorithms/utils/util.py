import copy
import numpy as np
import math
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as func


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def calculate_conv_params(input_size):
    """
    Compute the padding, stride, and kernel size for a given input image size.
    This function aims to preserve the spatial dimensions of the image.
   
    Args:
    - input_size (tuple): The shape of the input image in the format (height, width, channels).
   
    Returns:
    - tuple: (kernel_size, stride, padding)
    """

    # Assuming we want to keep the spatial dimensions same after convolution
    height, width, channels = input_size

    # Here we are making an assumption that if an image's dimensions are greater than a certain threshold,
    # we'd prefer a larger kernel size, otherwise, we'd stick to a smaller one.
    # You can adjust these heuristics based on your requirements.
    if height > 100 or width > 100:
        kernel_size = 5
    else:
        kernel_size = 3

    stride = 1  # To keep spatial dimensions same, stride should be 1

    # Padding is calculated based on the formula to preserve spatial dimensions:
    # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
    # Since we want output_size to be same as input_size (for stride=1), padding is:
    padding = (kernel_size - 1) // 2

    return kernel_size, stride, padding


def calculate_init(
        num_res_layers,
        output_change_scale='O(logn)',
) -> int:
    r"""
    Calculate initialization for omega.

    Parameters
    ----------
    num_res_layers: ``int``, required.
        The total number of residual layers. Typical n-layer Transformer encoder has 2n residual layers.
    output_change_scale: ``str``, optional (default = ``'O(logn)'``).
        The desired output change scale at initialization. Only ``'O(n)'``, ``'O(logn)'`` / ``'default'``,
        and ``'O(1)'`` are supported.

    Returns
    -------
    int: It would return the initialization value.
    """
    if 'O(logn)' == output_change_scale or 'default' == output_change_scale:
        omega_value = (num_res_layers + 1) / math.log(num_res_layers + 1) - 1
    elif 'O(n)' == output_change_scale:
        omega_value = 1.
    else:
        assert 'O(1)' == output_change_scale, \
            'only O(n), O(logn), and O(1) output changes are supported.'
        omega_value = num_res_layers
    return omega_value ** 0.5


def as_parameter(
        network,
        parameter_name,
        num_res_layers,
        embed_dim,
        output_change_scale='default',
) -> None:
    omega_vector = torch.ones(embed_dim)
    omega_vector.data.fill_(calculate_init(num_res_layers, output_change_scale))
    network.register_parameter(parameter_name, torch.nn.Parameter(omega_vector))


def _entropy(p):
    return -torch.sum(p * torch.log(p + 1e-9), dim=-1).mean()


def log_info(use_wandb, agent_name, info_key, info_value, counter, writer=None):
    agent_k = info_key + "/" + agent_name
    if use_wandb:
        wandb.log({agent_k: info_value}, step=counter)
    else:
        writer.add_scalars(agent_k, {agent_k: info_value}, counter)


class global_step_counter:
    def __init__(self):
        self.current_ep = 0

    def increment(self):
        self.current_ep += 1

    def cur_ep(self):
        return self.current_ep
