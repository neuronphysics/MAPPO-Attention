import copy
import numpy as np

import torch
import torch.nn as nn
import math
import os
from datetime import timedelta
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
import torch.distributed as dist
import json
from collections import defaultdict


def get_optimizer_groups(model, args):
    """Enhanced parameter grouping with explicit base encoder handling"""
    params = []
    param_groups = {
        'slot_lora': [],
        'slot': [],
        'layernorm': [],
        'other': [],
        'frozen': []
    }

    # Categorize parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param_groups['frozen'].append(param)
            continue
        if 'slot_attn' in name and (('norm_' in name) or ('layernorm' in name.lower()) or ('.mlp.1.' in name)):
            param_groups['layernorm'].append(param)
            
        elif 'slot_attn' in name:
            if 'lora_' in name:
                param_groups['slot_lora'].append(param)
            else:
                param_groups['slot'].append(param)
        else:
            param_groups['other'].append(param)

    print("\nParameter Group Statistics:")
    for group_name, group_params in param_groups.items():
        if group_name == 'frozen':
            total_params = sum(p.numel() for p in group_params)
            print(f"Frozen parameters: {total_params} parameters")
        else:
            total_params = sum(p.numel() for p in group_params)
            print(f"{group_name}: {len(group_params)} layers, {total_params} parameters")

    # Create parameter groups (rest remains the same)
    lr_mapping = {
        'slot_lora': args.lr_main,
        'slot': args.lr_main,
        'layernorm': args.lr_main,
        'other': args.lr
    }
    clip_mapping = {
        'slot_lora': True,
        'slot': True,
        'layernorm': False,
        'other': False
    }
    ewc_lambda_mapping = {
        'slot_lora': args.ewc_lambda,
        'slot': args.ewc_lambda,
        'layernorm': 0.0,  # No EWC regularization for layernorm parameters
        'other': 0.0  # No EWC regularization for other parameters
    }
    ewc_beta_mapping = {
        'slot_lora': args.ewc_beta_weight,
        'slot': args.ewc_beta_weight,
        'layernorm': 0.0,  # No EWC regularization for layernorm parameters
        'other': 0.0  # No EWC regularization for other parameters
    }
    ewc_beta_fisher_mapping = {
        'slot_lora': args.ewc_beta_fisher,
        'slot': args.ewc_beta_fisher,
        'layernorm': 0.0,  # No EWC regularization for layernorm parameters
        'other': 0.0  # No EWC regularization for other parameters
    }
    for group_type in ['slot_lora', 'slot', 'layernorm', 'other']:
        if param_groups[group_type]:
            params.append({
                'params': param_groups[group_type],
                'lr': lr_mapping[group_type],
                'name': group_type,
                'needs_clipping': clip_mapping[group_type],
                'beta': args.weight_clip_beta if clip_mapping[group_type] else 0.0,
                'ewc_lambda': ewc_lambda_mapping[group_type],
                'ewc_beta_weight': ewc_beta_mapping[group_type],
                'ewc_beta_fisher': ewc_beta_fisher_mapping[group_type],
            })

    total_trainable = sum(p.numel() for group in params for p in group['params'])
    if total_trainable == 0:
        raise RuntimeError("No trainable parameters - check fine-tuning config")
        
    return params

class InitBounds:
    '''
    A class to calculate the initial bounds for weight clipping.
    Uniform Kaiming initialization bounds are used.
    Since bias requires knowledge of the previous layer's weights, we keep track of the previous weight tensor in this class.
    Linear: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L106
    Conv2d: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L144
    '''
    def __init__(self):
        self.previous_weight = None

    def get(self, p):
        if p.dim() == 1:
            if self.previous_weight is not None:
               fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.previous_weight)
            else:
                fan_in = p.size(0)
            return 1.0 / math.sqrt(fan_in)
        elif p.dim() == 2 or p.dim() == 4:
            self.previous_weight = p
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(p)
            return  1.0 / math.sqrt(fan_in)
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(p.dim()))
        
class EWCWeightClipping(torch.optim.Optimizer):
    def __init__(self, params, pretrained_weights=None, beta=1.0, ewc_lambda=0.01, ewc_beta_weight=0.999, ewc_beta_fisher=0.999, optimizer=torch.optim.Adam, **kwargs):
        defaults = dict(beta=beta, ewc_lambda=ewc_lambda, ewc_beta_weight=ewc_beta_weight, ewc_beta_fisher=ewc_beta_fisher)
        super(EWCWeightClipping, self).__init__(params, defaults)
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.defaults.update(self.optimizer.defaults)
        self.init_bounds = InitBounds()
        # Store pretrained weights reference
        self.pretrained_weights = pretrained_weights or {}
        
        # Create mapping between parameters and names
        self.param_name_map = {}
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                    
                # Find parameter name by comparing with stored parameters
                for name, saved_param in self.pretrained_weights.items():
                    # Compare shape and some values to identify matching parameters
                    if p.shape == saved_param.shape and torch.all(p[:5] == saved_param[:5].to(p.device)):
                        self.param_name_map[p.data_ptr()] = name
                        break
 

    def step(self):
        self._apply_ewc()
        self.optimizer.step()
        self.weight_clipping()

    def weight_clipping(self):
        for group in self.param_groups:
            if not group['needs_clipping']:
                continue
                
            for p in group['params']:
                if not p.requires_grad: # Skip parameters that weren't trained
                    continue

                bound = self.init_bounds.get(p)
                if bound is not None:
                    p.data.clamp_(-group['beta'] * bound, group['beta'] * bound)

    def _apply_ewc(self):
        """
        EWC is applied by modifying gradients, not by directly updating parameters
        """
        for group in self.param_groups:
            # Only apply EWC to slot and slot_lora parameters
            if group['name'] not in ['slot', 'slot_lora']:
                continue
                
            for p in group["params"]:
                if not p.requires_grad or p.grad is None:
                    continue
                    
                state = self.state[p]
                param_name = self.param_name_map.get(p.data_ptr())
                if len(state) == 0:  # Safety check
                    state["ewc_step"] = 0
#                    state["weight_trace"] = torch.zeros_like(p.data)
                    state["fisher_trace"] = torch.zeros_like(p.data)
                    # Use pretrained weight if available for this parameter
                    if param_name and param_name in self.pretrained_weights:
                        state["original_weight"] = self.pretrained_weights[param_name].to(p.device)
                    else:
                        # Fallback to current weights if no pretrained weight found
                        state["original_weight"] = torch.zeros_like(p.data)
 
                state["ewc_step"] += 1
                
                weight_trace = state["original_weight"]
                fisher_trace = state["fisher_trace"]
                
                # Update EMA of weights and fisher information
                weight_trace.mul_(group["ewc_beta_weight"]).add_(p.data, alpha=1 - group["ewc_beta_weight"])
                fisher_trace.mul_(group["ewc_beta_fisher"]).add_(p.grad.data ** 2, alpha=1 - group["ewc_beta_fisher"])
                
                # Bias correction
                bias_correction_weight = 1 - group["ewc_beta_weight"] ** state["ewc_step"]
                bias_correction_fisher = 1 - group["ewc_beta_fisher"] ** state["ewc_step"]
                
                # Calculate consolidation term and add to gradient
                weight_consolidation = group["ewc_lambda"] * fisher_trace * (p.data - weight_trace / bias_correction_weight) / bias_correction_fisher
                p.grad.data.add_( weight_consolidation) 

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def print_cuda_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


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


class global_step_counter:
    def __init__(self):
        self.current_ep = 0

    def increment(self):
        self.current_ep += 1

    def get_cur_ep(self):
        return self.current_ep


def distributed_setup():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), dist.get_local_rank(), torch.device(
            f"cuda:{dist.get_local_rank()}")

    assert torch.distributed.is_available()
    print("PyTorch Distributed available.")
    print("  Backends:")
    print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    print(f"    MPI:  {torch.distributed.is_mpi_available()}")

    # NOTE: the env:// init method uses FileLocks, which sometimes causes deadlocks due to the
    # distributed filesystem configuration on the Mila cluster.
    # For multi-node jobs, use the TCP init method instead.
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    # timeout = timedelta(seconds=60)

    # DDP Job is being run via `srun` on a slurm cluster.
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    # SLURM var -> torch.distributed vars in case needed
    # NOTE: Setting these values isn't exactly necessary, but some code might assume it's
    # being run via torchrun or torch.distributed.launch, so setting these can be a good idea.
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=60)
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def cleanup_nccl():
    torch.distributed.destroy_process_group()


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


class ObsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def selectively_unfreeze_layers(model, target_modules):
    """
    Freeze all model parameters except for specified layers.

    Args:
        model: The model to modify
        target_modules: List of strings matching the layer names to unfreeze
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze specified layers
    for name, param in model.named_parameters():
        # Keep LayerNorms trainable
        if ('norm_' in name) or ('layernorm' in name.lower()) or ('.mlp.1.' in name):
            param.requires_grad = True
        # Check if the parameter name contains any of the target modules
        if any(target in name for target in target_modules):
            param.requires_grad = True
            print(f"Unfrozen layer: {name}")


