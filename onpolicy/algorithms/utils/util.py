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

class DormantNeuronTracker:
    def __init__(self, model, threshold=0.001):
        """
        Initialize the tracker for dormant neurons.
        Args:
            model (nn.Module): CLIP model instance.
            threshold (float): Threshold to define dormant neurons (default: 0.01).
        """
        self.model = model
        self.threshold = threshold
        self.dormant_neurons = {"activation": {}}
        self.dormant_neurons_weight = defaultdict(list)
        self.total_neurons = {"activation": 0}
        self.total_neuron = 0


    def clear_activation_data(self):
        """
        Clear all activation-based dormant neuron tracking data.
        """
        self.dormant_neurons["activation"].clear()
        self.total_neurons["activation"] = 0

    ### Activation-Based Tracking ###
    def initialize_total_neurons(self):
        """
        Compute and store the total number of neurons for all layers being tracked.
        """
        total_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
                # Count neurons based on the number of output features (weight.shape[0])
                if hasattr(module, "weight") and module.weight is not None:
                    is_trainable = module.weight.requires_grad
                    has_lora = any('lora' in n for n, p in module.named_parameters())
                    
                    if is_trainable or has_lora:
                        # Handle different layer types
                        if isinstance(module, nn.Embedding):
                            num_neurons = module.embedding_dim  # CORRECT: embedding dimension
                        elif isinstance(module, nn.Conv2d):
                            num_neurons = module.out_channels
                        elif isinstance(module, nn.Linear):
                            num_neurons = module.out_features
                        elif isinstance(module, nn.LayerNorm):
                            num_neurons = module.normalized_shape[0]
                        total_count += num_neurons  # CORRECT COUNTING
                       
        self.total_neurons["activation"] = total_count
        self.total_neuron = total_count
        print(f"Initialized Total Neurons: {total_count}")
        del total_count, is_trainable, has_lora 

    def register_activation_hooks(self):
        """
        Register hooks to capture activations across all layers, including embeddings.
        """
        for name, module in self.model.named_modules():
            # Ensure relevant modules like Linear, Conv2d, Embedding, and LayerNorm are covered
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
                if hasattr(module, "weight") and (module.weight.requires_grad or any('lora' in n for n, p in module.named_parameters())):
                   module.register_forward_hook(self._create_activation_hook(name))

    def _create_activation_hook(self, layer_name):
        """
        Internal method to create a hook for tracking activations.
        """
        def hook(module, input, output):
            # Skip if layer is frozen and not LoRA-tuned
            if not module.weight.requires_grad and not any('lora' in n for n, p in module.named_parameters()):
                return
            with torch.no_grad():
                if isinstance(output, tuple):
                    output = output[0]
                if isinstance(module, nn.Conv2d):
                    mean_activation = output.mean(dim=[0, 2, 3])  # Batch and spatial dimensions
                elif isinstance(module, nn.Embedding):
                    mean_activation = output.mean(dim=[0,1])
                else:
                    # Mean over ALL dimensions except last (features)
                    mean_activation = output.mean(dim=tuple(range(output.ndim - 1)))
                dormant_indices = (mean_activation < self.threshold).nonzero(as_tuple=True)[0].tolist()
                self.dormant_neurons["activation"][layer_name] = dormant_indices
                # Clean up
                del mean_activation, dormant_indices
                
        return hook

    ### Unified Helper Functions ###
    def calculate_dormant_ratio(self, mode):
        """
        Calculate the ratio of dormant neurons based on the tracking mode.
        Args:
            mode (str): Either "activation" or "weight_update".
        """
        if mode not in self.dormant_neurons:
            raise ValueError(f"Invalid mode: {mode}. Choose 'activation' or 'weight_update'.")
        dormant_count = 0
        if mode == "weight_update":
            dormant_count = 0
            for name, indices in self.dormant_neurons_weight.items():
                # `indices` are the dormant neuron indices for the given layer
                dormant_count += len(indices)  # Count dormant neurons
            total_count = self.total_neuron
            print(f"Dormant Count: {dormant_count}, Total Neuron Count: {total_count}")
            return dormant_count / total_count if total_count > 0 else 0
        else:# mode == "activation"
            dormant_count = sum(len(indices) for indices in self.dormant_neurons[mode].values())
            total_count = self.total_neuron
            print(f"Dormant Count: {dormant_count}, Total Neuron Count: {total_count}")
            ratio = dormant_count / total_count if total_count > 0 else 0
            assert ratio <= 1.0, "Dormant ratio exceeds 1.0"
            return ratio

    def save(self, path, mode):
        """
        Save the tracked dormant neurons to a JSON file.
        Args:
            path (str): File path to save the JSON data.
            mode (str): Either "activation" or "weight_update".
        """
        if mode not in self.dormant_neurons:
            raise ValueError(f"Invalid mode: {mode}. Choose 'activation' or 'weight_update'.")

        with open(path, "w") as f:
            json.dump(self.dormant_neurons[mode], f, indent=4)

    def load(self, path, mode):
        """
        Load dormant neuron data from a JSON file.
        Args:
            path (str): File path to load the JSON data.
            mode (str): Either "activation" or "weight_update".
        """
        if mode not in self.dormant_neurons:
            raise ValueError(f"Invalid mode: {mode}. Choose 'activation' or 'weight_update'.")

        with open(path, "r") as f:
            self.dormant_neurons[mode] = json.load(f)

    ### Verification and Debugging ###
    def verify_all_hooks(self):
        """
        Verify if all relevant layers have hooks registered.
        """
        registered_layers = list(self.dormant_neurons["activation"].keys())
        print(f"Total layers with activation hooks: {len(registered_layers)}")
        print(f"Sample layers: {registered_layers[:10]}")

    def print_model_structure(self):
        """
        Print the model structure for debugging and verification purposes.
        """
        print("Listing all layers in the model:")
        for name, module in self.model.named_modules():
            print(f"{name}: {module}")




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
        # Check if the parameter name contains any of the target modules
        if any(target in name for target in target_modules):
            param.requires_grad = True
            print(f"Unfrozen layer: {name}")
