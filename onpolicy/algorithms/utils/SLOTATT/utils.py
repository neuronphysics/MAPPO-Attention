import copy
import logging
import math
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Tuple, Union, Dict
import torch
from omegaconf import ListConfig, DictConfig
from torch import Tensor, nn
import yaml
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from ignite.engine import Engine, Events
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from math import sqrt
from ignite.engine.events import CallableEventWithFilter
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import sys
from ignite.contrib.handlers import ProgressBar
from sklearn.metrics import adjusted_rand_score
from itertools import chain
from .CoordConv import CoordConv2d, CoordConvTranspose2d

MANDATORY_FIELDS = [
    "loss",  # training loss
    "mask",  # masks for all slots (incl. background if any)
    "slot",  # raw slot reconstructions for all slots (incl. background if any)
    "representation",  # slot representations (only foreground, if applicable)
]


@dataclass(eq=False, repr=False)
class BaseModel(nn.Module):
    name: str
    width: int
    height: int

    # This applies only to object-centric models, but must always be defined.
    num_slots: int

    def __post_init__(self):
        # Run the nn.Module initialization logic before we do anything else. Models
        # should call this post-init at the beginning of their post-init.
        super().__init__()

    @property
    def num_representation_slots(self) -> int:
        """Number of slots used for representation.

        By default, it is equal to the number of slots, but when possible we can
        consider only foreground slots (e.g. in SPACE).
        """
        return self.num_slots

    @property
    @abstractmethod
    def slot_size(self) -> int:
        """Representation size per slot.

        This does not apply to models that are not object-centric, but they should still
        define it in the most sensible possible way.
        """
        ...


@dataclass
class ForwardPass:
    model: BaseModel
    device: Union[torch.device, str]
    preprocess_fn: Optional[Callable] = None

    def __call__(self, batch: dict, mode: str = "train") -> Tuple[dict, dict]:
        for key in batch.keys():
            batch[key] = batch[key].to(self.device, non_blocking=True)
        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self.model(batch["image"])

        if mode == 'train':
            self.model.train_discriminator(batch["image"])

        return batch, output


def extract_state_dicts(state: dict) -> dict:
    return {name: state[name].state_dict() for name in state}


@torch.no_grad()
def init_trunc_normal_(model: nn.Module, mean: float = 0.0, std: float = 1.0):
    """Initializes (in-place) a model's weights with truncated normal, and its biases to zero.

    All parameters with name ending in ".weight" are initialized with a truncated
    normal distribution with specified mean and standard deviation. The truncation
    is at plus/minus 2 stds from the mean.

    All parameters with name ending in ".bias" are initialized to zero.

    Args:
        model: The model.
        mean: Mean of the truncated normal.
        std: Standard deviation of the truncated normal.
    """
    for name, tensor in model.named_parameters():
        if name.endswith(".bias"):
            tensor.zero_()
        elif name.endswith(".weight"):
            nn.init.trunc_normal_(tensor, mean, std, a=mean - 2 * std, b=mean + 2 * std)


@torch.no_grad()
def init_xavier_(model: nn.Module):
    """Initializes (in-place) a model's weights with xavier uniform, and its biases to zero.

    All parameters with name containing "bias" are initialized to zero.

    All other parameters are initialized with xavier uniform with default parameters,
    unless they have dimensionality <= 1.

    Args:
        model: The model.
    """
    for name, tensor in model.named_parameters():
        if name.endswith(".bias"):
            tensor.zero_()
        elif len(tensor.shape) <= 1:
            pass  # silent
        else:
            torch.nn.init.xavier_uniform_(tensor)


def get_activation_module(activation_name: str, try_inplace: bool = True) -> nn.Module:
    if activation_name == "leakyrelu":
        act = torch.nn.LeakyReLU()
    elif activation_name == "elu":
        act = torch.nn.ELU()
    elif activation_name == "relu":
        act = torch.nn.ReLU(inplace=try_inplace)
    elif activation_name == "glu":
        act = torch.nn.GLU(dim=1)  # channel dimension in images
    elif activation_name == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activation_name == "tanh":
        act = torch.nn.Tanh()
    else:
        raise ValueError(f"Unknown activation name '{activation_name}'")
    return act


def get_conv_output_shape(
        width: int,
        height: int,
        kernels: List[int],
        paddings: List[int],
        strides: List[int],
) -> Tuple[int, int]:
    for kernel, stride, padding in zip(kernels, strides, paddings):
        width = (width + 2 * padding - kernel) // stride + 1
        height = (height + 2 * padding - kernel) // stride + 1
    return width, height


def summary_num_params(
        model: nn.Module, max_depth: Optional[int] = 4
) -> Tuple[str, int]:
    """Generates overview of the number of parameters in each component of the model.

    Optionally, it groups together parameters below a certain depth in the
    module tree.

    Args:
        model (torch.nn.Module)
        max_depth (int, optional)

    Returns:
        tuple: (summary string, total number of trainable parameters)
    """

    sep = "."  # string separator in parameter name
    out = "\n--- Trainable parameters:\n"
    num_params_tot = 0
    num_params_dict = OrderedDict()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_params = param.numel()

        if max_depth is not None:
            split = name.split(sep)
            prefix = sep.join(split[:max_depth])
        else:
            prefix = name
        if prefix not in num_params_dict:
            num_params_dict[prefix] = 0
        num_params_dict[prefix] += num_params
        num_params_tot += num_params
    for n, n_par in num_params_dict.items():
        out += f"{n_par:8d}  {n}\n"
    out += f"  - Total trainable parameters: {num_params_tot}\n"
    out += "---------\n\n"

    return out, num_params_tot


def grad_global_norm(
        parameters: Union[Iterable[Tensor], Tensor],
        norm_type: Optional[Union[float, int]] = 2,
) -> float:
    """Computes the global norm of the gradients of an iterable of parameters.

    The norm is computed over all gradients together, as if they were concatenated
    into a single vector.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor
        norm_type (float or int, optional): type of the used p-norm. Can be
            ``'inf'`` for infinity norm.

    Returns:
        Global norm of the parameters' gradients (viewed as a single vector).
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    return global_norm(grads, norm_type=norm_type)


def global_norm(
        parameters: Union[Iterable[Tensor], Tensor],
        norm_type: Optional[Union[float, int]] = 2,
) -> float:
    """Computes the global norm of an iterable of parameters.

    The norm is computed over all tensors together, as if they were concatenated
    into a single vector. This code is based on torch.nn.utils.clip_grad_norm_().

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor
        norm_type (float or int, optional): type of the used p-norm. Can be
            ``'inf'`` for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0.0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm **= 1.0 / norm_type
    return total_norm


def _apply_to_param_group(fn: Callable, model: nn.Module, group_name: str):
    return fn([x[1] for x in model.named_parameters() if x[0].startswith(group_name)])


def group_grad_global_norm(model: nn.Module, group_name: str) -> float:
    """Returns the global norm of the gradiends of a group of parameters in the model.

    Args:
        model: The model.
        group_name: The group name.

    Returns:
        The global norm of the gradients of a group of parameters in the model
        whose name starts with `group_name`.
    """
    return _apply_to_param_group(grad_global_norm, model, group_name)


def group_global_norm(model: nn.Module, group_name: str) -> float:
    """Returns the global norm of a group of parameters in the model.

    Args:
        model: The model.
        group_name: The group name.

    Returns:
        The global norm of the group of parameters in the model
        whose name starts with `group_name`.
    """
    return _apply_to_param_group(global_norm, model, group_name)


def _scalars_to_list(params: dict) -> dict:
    # Channels must be a list
    list_size = len(params["channels"])
    # All these must be in `params` and should be expanded to list
    allow_list = [
        "kernels",
        "batchnorms",
        "bn_affines",
        "paddings",
        "strides",
        "activations",
        "output_paddings",
        "conv_transposes",
    ]
    for k in allow_list:
        if not isinstance(params[k], (tuple, list, ListConfig)):
            params[k] = [params[k]] * list_size
    return params


def make_sequential_from_config(
        input_channels: int,
        channels: List[int],
        kernels: Union[int, List[int]],
        batchnorms: Union[bool, List[bool]],
        bn_affines: Union[bool, List[bool]],
        paddings: Union[int, List[int]],
        strides: Union[int, List[int]],
        activations: Union[str, List[str]],
        output_paddings: Union[int, List[int]] = 0,
        conv_transposes: Union[bool, List[bool]] = False,
        return_params: bool = False,
        try_inplace_activation: bool = True,
) -> Union[nn.Sequential, Tuple[nn.Sequential, dict]]:
    # Make copy of locals and expand scalars to lists
    params = {k: v for k, v in locals().items()}
    params = _scalars_to_list(params)

    # Make sequential with the following order:
    # - Conv or conv transpose
    # - Optional batchnorm (optionally affine)
    # - Optional activation
    layers = []
    layer_infos = zip(
        params["channels"],
        params["batchnorms"],
        params["bn_affines"],
        params["kernels"],
        params["strides"],
        params["paddings"],
        params["activations"],
        params["conv_transposes"],
        params["output_paddings"],
    )
    for (
            channel,
            bn,
            bn_affine,
            kernel,
            stride,
            padding,
            activation,
            conv_transpose,
            o_padding,
    ) in layer_infos:
        if conv_transpose:
            layers.append(
                CoordConvTranspose2d(
                    input_channels, channel, kernel, stride, padding, o_padding
                )
            )
        else:

            layers.append(CoordConv2d(input_channels, channel, kernel, stride, padding))

        if bn:
            layers.append(nn.BatchNorm2d(channel, affine=bn_affine))
        if activation is not None:
            layers.append(
                get_activation_module(activation, try_inplace=try_inplace_activation)
            )

        # Input for next layer has half the channels of the current layer if using GLU.
        input_channels = channel
        if activation == "glu":
            input_channels //= 2

    if return_params:
        return nn.Sequential(*layers), params
    else:
        return nn.Sequential(*layers)


def log_residual_stack_structure(
        channel_size_per_layer: List[int],
        layers_per_block_per_layer: List[int],
        downsample: int,
        num_layers_per_resolution: List[int],
        encoder: bool = True,
) -> List[str]:
    logging.debug(f"Creating structure with {downsample} downsamples.")
    out = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            out.append(
                "Residual Block with "
                "{} channels and "
                "{} layers.".format(
                    channel_size_per_layer[layer], layers_per_block_per_layer[layer]
                )
            )
            layer += 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    out.append(
                        "Con2d layer with "
                        "{} input channels and "
                        "{} output channels".format(
                            channel_size_per_layer[layer - 1],
                            channel_size_per_layer[layer],
                        )
                    )
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

        # after the residual block, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                out.append("Avg Pooling layer.")
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                out.append("Interpolation layer.")

    return out


def build_residual_stack(
        channel_size_per_layer: List[int],
        layers_per_block_per_layer: List[int],
        downsample: int,
        num_layers_per_resolution: List[int],
        encoder: bool = True,
) -> List[nn.Module]:
    logging.debug(
        "\n".join(
            log_residual_stack_structure(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=encoder,
            )
        )
    )
    layers = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            # add a residual block with the required number of channels and layers
            layers.append(
                ResidualBlock(
                    channel_size_per_layer[layer],
                    num_layers=layers_per_block_per_layer[layer],
                )
            )
            layer += 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

                    in_channels = channel_size_per_layer[layer - 1]
                    out_channels = channel_size_per_layer[layer]
                    layers.append(CoordConv2d(in_channels, out_channels, kernel_size=1))

        # after the residual blocks, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                layers.append(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                layers.append(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )

    return layers


def single_forward_pass(
        model: BaseModel, dataloader: DataLoader, device: Union[torch.device, str]
) -> Tuple[dict, dict]:
    eval_step = ForwardPass(model, device)
    evaluator = Engine(lambda e, b: eval_step(b))
    evaluator.run(dataloader, 1, 1)
    batch, output = evaluator.state.output
    return batch, output


class TrainCheckpointHandler:
    def __init__(
            self, checkpoint_path: Union[str, Path], device: Union[torch.device, str]
    ):
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.checkpoint_train_path = checkpoint_path / "train_checkpoint.pt"
        self.model_path = checkpoint_path / "model.pt"
        self.train_yaml_path = checkpoint_path / "train_state.yaml"
        self.device = device

    def save_checkpoint(self, state_dicts: dict):
        """Saves a checkpoint.

        If the state contains the key "model", the model parameters are saved
        separately to model.pt, and they are not saved to the checkpoint file.
        """
        if "model" in state_dicts:
            logging.info(f"Saving model to {self.model_path}")
            torch.save(state_dicts["model"], self.model_path)
            del state_dicts["model"]  # do not include model (avoid duplicating)
        torch.save(state_dicts, self.checkpoint_train_path)

        # Save train state (duplicate info from main checkpoint)
        trainer_state = state_dicts["trainer"]
        with open(self.train_yaml_path, "w") as f:
            train_state = {
                "step": trainer_state["iteration"],
                "max_step": trainer_state["epoch_length"],
            }
            yaml.dump(train_state, f)

    def load_checkpoint(self, objects: dict):
        """Loads checkpoint into the provided dictionary."""

        # Load checkpoint without model
        state = torch.load(self.checkpoint_train_path, self.device)
        for varname in state:
            logging.debug(f"Loading checkpoint: variable name '{varname}'")
            objects[varname].load_state_dict(state[varname])

        # Load model
        if "model" in objects:
            logging.debug(f"Loading checkpoint: model")
            model_state_dict = torch.load(self.model_path, self.device)
            objects["model"].load_state_dict(model_state_dict)


def linear_warmup_exp_decay(
        warmup_steps: Optional[int] = None,
        exp_decay_rate: Optional[float] = None,
        exp_decay_steps: Optional[int] = None,
) -> Callable[[int], float]:
    assert (exp_decay_steps is None) == (exp_decay_rate is None)
    use_exp_decay = exp_decay_rate is not None
    if warmup_steps is not None:
        assert warmup_steps > 0

    def lr_lambda(step):
        multiplier = 1.0
        if warmup_steps is not None and step < warmup_steps:
            multiplier *= step / warmup_steps
        if use_exp_decay:
            multiplier *= exp_decay_rate ** (step / exp_decay_steps)
        return multiplier

    return lr_lambda


def infer_model_type(model_name: str) -> str:
    if model_name.startswith("baseline_vae"):
        return "distributed"
    if model_name in [
        "slot-attention",
        "monet",
        "genesis",
        "space",
        "monet-big-decoder",
        "slot-attention-big-decoder",
    ]:
        return "object-centric"
    raise ValueError(f"Could not infer model type for model '{model_name}'")


class PositionalEmbedding(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        east = torch.linspace(0, 1, width).repeat(height)
        west = torch.linspace(1, 0, width).repeat(height)
        south = torch.linspace(0, 1, height).repeat(width)
        north = torch.linspace(1, 0, height).repeat(width)
        east = east.reshape(height, width)
        west = west.reshape(height, width)
        south = south.reshape(width, height).T
        north = north.reshape(width, height).T
        # (4, h, w)
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, channels, kernel_size=1)
        self.register_buffer("linear_position_embedding", linear_pos_embedding)

    def forward(self, x: Tensor) -> Tensor:
        bs_linear_position_embedding = self.linear_position_embedding.expand(
            x.size(0), 4, x.size(2), x.size(3)
        )
        x = x + self.channels_map(bs_linear_position_embedding)
        return x


class ResidualBlock(nn.Module):
    def __init__(
            self,
            n_channels,
            *,
            num_layers=2,
            kernel_size=3,
            dilation=1,
            groups=1,
            rezero=True,
    ):
        super().__init__()
        ch = n_channels
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        layers = []
        for i in range(num_layers):
            layers.extend(
                [
                    nn.LeakyReLU(1e-2),
                    CoordConv2d(
                        ch,
                        ch,
                        kernel_size=kernel_size,
                        padding=pad,
                        dilation=dilation,
                        groups=groups,
                    ),
                ]
            )
        self.net = nn.Sequential(*layers)
        if rezero:
            self.gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.gate = 1.0

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.net(inputs) * self.gate


@dataclass
class TBLogger:
    working_dir: Path
    trainer: Engine
    model: nn.Module
    loss_terms: List[str]
    scalar_params: List[str]  # names of scalar parameters in output
    event_images: CallableEventWithFilter
    event_loss: CallableEventWithFilter
    event_stats: CallableEventWithFilter
    event_parameters: CallableEventWithFilter
    param_groups: Optional[List[str]] = None
    num_images: int = 3

    def __post_init__(self):
        self.writer = SummaryWriter(str(self.working_dir))

        # Forward methods directly to wrapped writer.
        self.add_scalar = self.writer.add_scalar
        self.add_image = self.writer.add_image
        self.add_images = self.writer.add_images

        # Attach logging events to trainer.
        add_event = self.trainer.add_event_handler
        add_event(self.event_images, self._log_images)
        add_event(self.event_loss, self._log_train_losses)
        add_event(self.event_loss, self._log_scalar_params)
        add_event(self.event_stats, self._log_stats)
        add_event(self.event_parameters, self._log_params)
        add_event(self.event_parameters, self._log_grouped_params)

    @torch.no_grad()
    def log_dict(self, metrics: dict, iteration_num: int, group_name: str):
        for metric_name in metrics:
            self.add_scalar(
                f"{group_name}/{metric_name}", metrics[metric_name], iteration_num
            )

    @torch.no_grad()
    def _log_images(self, engine):
        n_img = self.num_images
        batch, out = engine.state.output
        step = engine.state.iteration
        recon_img = make_recon_img(out["slot"][:n_img], out["mask"][:n_img])
        sqrt_nrow = int(sqrt(n_img))
        x = batch["image"][:n_img]
        assert len(out["mask"].shape) == 5  # B, slots, 1, H, W

        x_recon = _flatten_slots(torch.stack([x, recon_img], dim=1), nrow=sqrt_nrow)
        self.add_image("input-reconstruction", x_recon.clamp(0.0, 1.0), step)

        slot = _flatten_slots(out["slot"][:n_img], sqrt_nrow)
        self.add_image("slot", slot.clamp(0.0, 1.0), step)

        if "mask" in batch:
            flat_mask = _flatten_slots(batch["mask"][:n_img], sqrt_nrow)
            mask = make_grid(flat_mask, nrow=sqrt_nrow).float()
            self.add_image("mask: true", mask, step)

        masked_img = x.unsqueeze(1) * out['mask'][:n_img]
        flat_masked_img = _flatten_slots(masked_img, sqrt_nrow)
        mask_img_grid = make_grid(flat_masked_img, nrow=sqrt_nrow).float()
        self.add_image("mask x image", mask_img_grid, step)

        flat_pred_mask = _flatten_slots(out["mask"][:n_img], sqrt_nrow)
        pred_mask = make_grid(flat_pred_mask, nrow=sqrt_nrow)
        self.add_image("mask: pred", pred_mask, step)

        mask_segmap, pred_mask_segmap = _compute_segmentation_mask(batch, n_img, out)
        if mask_segmap is not None:
            self.add_images("segmentation: true", mask_segmap, step)
        self.add_images("segmentation: pred", pred_mask_segmap, step)

    @torch.no_grad()
    def _log_train_losses(self, engine):
        batch, output = engine.state.output
        self.log_dict(
            filter_dict(output, allow_list=self.loss_terms, inplace=False),
            engine.state.iteration,
            "train losses",
        )

    @torch.no_grad()
    def _log_scalar_params(self, engine):
        batch, output = engine.state.output
        self.log_dict(
            filter_dict(output, allow_list=self.scalar_params, inplace=False),
            engine.state.iteration,
            "model params",
        )

    @torch.no_grad()
    def _log_stats(self, engine):
        batch, output = engine.state.output
        for metric_name in output:
            if metric_name in self.loss_terms:  # already logged in _log_train_losses()
                continue
            if (
                    metric_name in self.scalar_params
            ):  # already logged in _log_scalar_params()
                continue
            prefix = "model outputs"
            if metric_name not in MANDATORY_FIELDS:
                prefix += f" ({self.model.name})"
            self._log_tensor(engine, f"{prefix}/{metric_name}", output[metric_name])

    @torch.no_grad()
    def _log_params(self, engine):
        """Logs the global norm of all parameters and of their gradients."""
        self.add_scalar(
            "param grad norms/global",
            grad_global_norm(self.model.parameters()),
            engine.state.iteration,
        )
        self.add_scalar(
            "param norms/global",
            global_norm(self.model.parameters()),
            engine.state.iteration,
        )

    @torch.no_grad()
    def _log_grouped_params(self, engine):
        """Logs the global norm of parameters and their gradients, by group."""
        if self.param_groups is None:
            return
        assert isinstance(self.param_groups, list)
        for name in self.param_groups:
            self.add_scalar(
                f"param grad norms/group: {name}",
                group_grad_global_norm(self.model, name),
                engine.state.iteration,
            )
            self.add_scalar(
                f"param norms/group: {name}",
                group_global_norm(self.model, name),
                engine.state.iteration,
            )

    @torch.no_grad()
    def _log_tensor(self, engine, name, tensor):
        if not isinstance(tensor, Tensor):
            return
        if tensor.numel() == 1:
            stats = ["item"]
        else:
            stats = ["min", "max", "mean"]
        for stat in stats:
            value = getattr(tensor, stat)()
            if stat == "item":
                name_ = name
            else:
                name_ = f"{name} [{stat}]"
            self.add_scalar(name_, value, engine.state.iteration)


def _flatten_slots(images: Tensor, nrow: int):
    image_lst = images.split(1, dim=0)
    image_lst = [
        make_grid(image.squeeze(0), nrow=images.shape[1]) for image in image_lst
    ]
    images = torch.stack(image_lst, dim=0)
    pad_value = 255 if isinstance(images, torch.LongTensor) else 1.0
    return make_grid(images, nrow=nrow, pad_value=pad_value, padding=4)


def _compute_segmentation_mask(batch, num_images, output):
    if "mask" in batch:
        # [bs, ns, 1, H, W] to [bs, 1, H, W]
        mask_segmap = batch["mask"][:num_images].argmax(1)
        # If shape is [bs, H, W], turn it into [bs, 1, H, W]
        if mask_segmap.shape[1] != 1:
            mask_segmap = mask_segmap.unsqueeze(1)

        mask_segmap = apply_color_map(mask_segmap)
    else:
        mask_segmap = None

    # [bs, ns, 1, H, W] to [bs, 1, H, W]
    pred_mask_segmap = output["mask"][:num_images].argmax(1)

    # If shape is [bs, H, W], turn it into [bs, 1, H, W]
    if pred_mask_segmap.shape[1] != 1:
        pred_mask_segmap = pred_mask_segmap.unsqueeze(1)

    pred_mask_segmap = apply_color_map(pred_mask_segmap)
    return mask_segmap, pred_mask_segmap


def filter_dict(
        d: dict,
        allow_list: Optional[List] = None,
        block_list: Optional[List] = None,
        inplace: bool = True,
        strict_allow_list: bool = True,
) -> dict:
    """Filters a dictionary based on its keys.

    If a block list is given, the keys in this list are discarded and everything else
    is kept. If an allow list is given, only keys in this list are kept. In this case,
    if `strict_allow_list` is True, all keys in the allow list must be in the dictionary.
    Exactly one of `allow_list` and `block_list` must be given.

    Args:
        d:
        allow_list:
        block_list:
        inplace:
        strict_allow_list:

    Returns: the filtered dictionary.
    """
    if (allow_list is None) == (block_list is None):
        raise ValueError("Exactly one of `allow_list` and `block_list` must be None.")
    if inplace:
        out = d
    else:
        out = copy.copy(d)
    if block_list is not None:
        _dict_block_allow_list(out, block_list, is_allow_list=False)
    if allow_list is not None:
        if strict_allow_list:
            diff = set(allow_list).difference(d.keys())
            if len(diff) > 0:
                raise ValueError(
                    f"Some allowed keys are not in the dictionary, but strict_allow_list=True: {diff}"
                )
        _dict_block_allow_list(out, allow_list, is_allow_list=True)
    return out


def _dict_block_allow_list(d: dict, list_: List, *, is_allow_list: bool):
    """Deletes keys in-place."""
    for key in list(d.keys()):
        condition = key in list_
        if is_allow_list:
            condition = not condition
        if condition:
            try:
                del d[key]
            except KeyError:
                pass


DEFAULT_COLOR_MAP = torch.LongTensor(
    [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
        (255, 255, 255),
        (0, 0, 0),
    ]
)


def apply_color_map(image_categorical, color_map=None):
    """Applies a colormap to an image with categorical values.

    Args:
        image_categorical (Tensor): Tensor with shape (B, 1, H, W) and integer values in [0, N-1].
        color_map (Tensor): LongTensor with shape (N, 3) representing colormap in RGB in [0, 255].

    Returns:
        Image representing
    """
    # TODO redundant with masks_to_segmentation except that the input here is categorical.
    color_map = color_map or DEFAULT_COLOR_MAP
    input_shape = list(image_categorical.shape)
    assert len(input_shape) == 4 and input_shape[1] == 1, f"input shape = {input_shape}"
    out_shape = input_shape[:]
    out_shape[1] = 3
    dst_tensor = torch.zeros(*out_shape, dtype=image_categorical.dtype)
    for i in range(input_shape[0]):
        dst_tensor_i = color_map[image_categorical[i].cpu().long() % len(color_map)].squeeze()
        if dst_tensor_i.shape[0] != 3:
            dst_tensor_i = dst_tensor_i.permute(2, 0, 1)
        dst_tensor[i] = dst_tensor_i
    return dst_tensor


def make_recon_img(slot, mask):
    """Returns an image from composing slots (weighted sum) according to the masks.

    Args:
        slot (Tensor): The slot-wise images.
        mask (Tensor): The masks. These are weights that should sum to 1 along the
            slot dimension, but this is not enforced.

    Returns:
        The image resulting from a weighted sum of the slots using the masks as weights.
    """
    b, s, ch, h, w = slot.shape  # B, slots, 3, H, W
    assert mask.shape == (b, s, 1, h, w)  # B, slots, 1, H, W
    return (slot * mask).sum(dim=1)  # B, 3, H, W


_DEFAULT_METRICS = [
    "ari",
    "mean_segcover",
    "scaled_segcover",
    "mse",
    "mse_unmodified_fg",
    "mse_fg",
]


@dataclass
class MetricsEvaluator:
    dataloader: DataLoader
    loss_terms: List[str]
    skip_background: bool
    device: str
    metrics: List[str] = field(default_factory=lambda: _DEFAULT_METRICS)

    _forward_pass: ForwardPass = field(init=False)
    num_bg_objects: int = field(init=False)
    num_ignored_objects: int = field(init=False)

    def __post_init__(self):
        # This should be a MultiObjectDataset.
        dataset: MultiObjectDataset = self.dataloader.dataset  # type: ignore
        self.num_bg_objects = dataset.num_background_objects
        self.num_ignored_objects = self.num_bg_objects if self.skip_background else 0

    @torch.no_grad()
    def _eval_step(self, engine: Engine, batch: dict):
        batch, output = self._forward_pass(batch, mode="eval")

        # Compute metrics
        reconstruction = make_recon_img(output["slot"], output["mask"]).clamp(0.0, 1.0)
        mse_full = (batch["image"] - reconstruction) ** 2
        mse = mse_full.mean([1, 2, 3])

        if "mask" in batch:
            # One-hot to categorical masks
            true_mask = batch["mask"].cpu().argmax(dim=1)
            pred_mask = output["mask"].cpu().argmax(dim=1, keepdim=True).squeeze(2)

            if output["mask"].shape[1] == 1:  # not an object-centric model
                ari = mean_segcover = scaled_segcover = torch.full(
                    (true_mask.shape[0],), fill_value=torch.nan
                )
            else:
                # Num background objects should be equal (for each sample) to:
                # batch["visibility"].sum([1, 2]) - batch["num_actual_objects"].squeeze(1)
                ari = ari_(true_mask, pred_mask, self.num_ignored_objects)
                mean_segcover, scaled_segcover = segmentation_covering(
                    true_mask, pred_mask, self.num_ignored_objects
                )
        else:
            ari = None
            mean_segcover = None
            scaled_segcover = None

        if "is_foreground" in batch:
            # Mask shape (B, O, 1, H, W), is_foreground (B, O, 1), is_modified (B, O), where
            # O = max num objects. Expand the last 2 to (B, O, 1, 1, 1) for broadcasting.
            unsqueezed_shape = (*batch["is_foreground"].shape, 1, 1)
            is_fg = batch["is_foreground"].view(*unsqueezed_shape)
            is_modified = batch["is_modified"].view(*unsqueezed_shape)

            # Mask with foreground objects: shape (B, 1, H, W)
            fg_mask = (batch["mask"] * is_fg).sum(1)
            # Mask with unmodified foreground objects: shape (B, 1, H, W)
            unmodified_fg_mask = (batch["mask"] * is_fg * (1 - is_modified)).sum(1)

            # MSE computed only on foreground objects.
            fg_mse = (mse_full * fg_mask).mean([1, 2, 3])
            # MSE computed only on foreground objects that were not modified.
            unmodified_fg_mse = (mse_full * unmodified_fg_mask).mean([1, 2, 3])
        else:
            unmodified_fg_mse = None
            fg_mse = None

        # Collect loss values from model output
        loss_values = {}
        for loss_term in self.loss_terms:
            loss_values[loss_term] = output[loss_term]

        # for unlabeled dataset the there is only mse and loss value returned
        # Return with shape (batch_size, )
        return dict(
            ari=ari,
            mse=mse,
            mse_unmodified_fg=unmodified_fg_mse,
            mse_fg=fg_mse,
            mean_segcover=mean_segcover,
            scaled_segcover=scaled_segcover,
            **loss_values
        )

    @torch.no_grad()
    def eval(
            self, model: BaseModel, steps: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        self._forward_pass = ForwardPass(model, self.device)

        engine = Engine(self._eval_step)
        results = {name: [] for name in self.metrics + list(self.loss_terms)}

        @engine.on(Events.ITERATION_COMPLETED)
        def accumulate_metrics(engine):
            for name in self.metrics + list(self.loss_terms):
                batch_results = engine.state.output[name]
                if batch_results is None:
                    if name in results:
                        results.pop(name)
                    continue

                if batch_results.dim() == 0:  # scalar
                    batch_size = engine.state.batch["image"].shape[0]
                    batch_results = [batch_results] * batch_size
                results[name].extend(batch_results)

        if sys.stdout.isatty():
            ProgressBar().attach(
                engine,
                output_transform=lambda o: {k: v.mean() for k, v in o.items()},
            )
        engine.run(self.dataloader, 1, steps)

        # Split results into losses and metrics
        losses = {k: results[k] for k in self.loss_terms}
        metrics = {k: results[k] for k in self.metrics if k in results}

        return dict_tensor_mean(losses), dict_tensor_mean(metrics)


def dict_tensor_mean(data: Dict[str, List[Tensor]]) -> Dict[str, float]:
    output = {}
    for key, value in data.items():
        output[key] = torch.stack(value).mean().item()
    return output


@dataclass
class BaseTrainer:
    device: str
    steps: int
    optimizer_config: List[Dict]
    clip_grad_norm: Optional[float]
    checkpoint_steps: int
    logloss_steps: int
    logweights_steps: int
    logimages_steps: int
    logvalid_steps: int
    debug: bool
    resubmit_steps: Optional[int]
    resubmit_hours: Optional[float]
    working_dir: Path

    model: BaseModel = field(init=False)
    dataloaders: List[DataLoader] = field(init=False)
    optimizers: List[Optimizer] = field(init=False)
    trainer: Engine = field(init=False)
    evaluator: MetricsEvaluator = field(init=False)
    eval_step: ForwardPass = field(init=False)
    checkpoint_handler: TrainCheckpointHandler = field(init=False)
    lr_schedulers: List[LRScheduler] = field(init=False)  # optional schedulers
    training_start: float = field(init=False)

    def __post_init__(self):
        self.checkpoint_handler = TrainCheckpointHandler(self.working_dir, self.device)
        self.lr_schedulers = []  # No scheduler by default - subclasses append to this.

    def _make_optimizers(self, optim_config_list: List[Dict]):
        """Makes default optimizer on all model parameters.

        Called at the end of `_post_init()`. Override to customize.
        """

        # the optimizer config for base model is at 0 index
        model_optim_config = optim_config_list[0]
        alg = model_optim_config.pop("alg")  # In this base implementation, alg is required.
        opt_class = getattr(torch.optim, alg)
        self.optimizers = [opt_class(chain(self.model.encoder.parameters(), self.model.slot_attention.parameters(),
                                           self.model.decoder.parameters()), **model_optim_config)]

    def _setup_lr_scheduling(self):
        """Registers hook that steps all LR schedulers at each iteration.

        Called at the beginning of `_setup_training(). Override to customize.
        """

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def lr_scheduler_step(engine):
            logging.debug(f"Stepping {len(self.lr_schedulers)} schedulers")
            for scheduler in self.lr_schedulers:
                scheduler.step()

    @property
    def scalar_params(self) -> List[str]:
        """List of scalar model parameters that should be logged.

        They must be in the model's output dictionary. Empty list by default.
        """
        return []

    @property
    @abstractmethod
    def loss_terms(self) -> List[str]:
        ...

    @property
    def param_groups(self) -> List[str]:
        """Parameter groups whose norm and gradient norm will be logged separately to tensorboard."""
        return []

    def train_step(self, engine: Engine, batch: dict) -> Tuple[dict, dict]:
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        batch, output = self.eval_step(batch, mode="train")
        self._check_shapes(batch, output)  # check shapes of mandatory items
        output["loss"].backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm, error_if_nonfinite=True
            )
        for optimizer in self.optimizers:
            optimizer.step()
        return batch, output

    def _check_shapes(self, batch: dict, output: dict):
        bs = batch["image"].shape[0]
        if infer_model_type(self.model.name) == "distributed":
            n_slots = 1
            repr_shape = (bs, self.model.num_slots * self.model.slot_size)
        else:
            n_slots = self.model.num_slots
            repr_shape = (bs, self.model.num_representation_slots, self.model.slot_size)
        c = self.dataloaders[0].dataset.input_channels
        h, w = self.model.height, self.model.width
        # These are the fields in MANDATORY_FIELDS
        assert output["loss"].dim() == 0
        assert output["mask"].shape == (bs, n_slots, 1, h, w)
        assert output["slot"].shape == (bs, n_slots, c, h, w)
        assert output["representation"].shape == repr_shape

    def setup(
            self,
            model: BaseModel,
            dataloaders: List[DataLoader],
            load_checkpoint: bool = False,
    ):
        self._post_init(model, dataloaders)
        self._setup_training(load_checkpoint)

    def _post_init(self, model: BaseModel, dataloaders: List[DataLoader]):
        """Adds model and dataloaders to the trainer.

        Overriding methods should call this base method first.

        This method adds model and dataloaders to the Trainer object. It creates
        an evaluation step, the optimizer, and sets up tensorboard, but does not
        create a trainer engine. Anything that goes in the checkpoints must be
        created here. Anything that requires a trainer (e.g. callbacks) must be
        defined in `_setup_training()`.
        """
        assert model.training is True  # don't silently set it to train
        self.model = model
        self.dataloaders = dataloaders
        self.eval_step = ForwardPass(self.model, self.device)
        tensorboard_dir = self.working_dir + "tensorboard"
        self.trainer = Engine(self.train_step)
        self.logger = TBLogger(
            tensorboard_dir,
            self.trainer,
            model,
            loss_terms=self.loss_terms,
            scalar_params=self.scalar_params,
            event_images=Events.ITERATION_COMPLETED(every=self.logimages_steps),
            event_parameters=Events.ITERATION_COMPLETED(every=self.logweights_steps),
            event_loss=Events.ITERATION_COMPLETED(every=self.logloss_steps),
            event_stats=Events.ITERATION_COMPLETED(every=self.logloss_steps),
            param_groups=self.param_groups,
        )

        # Here we only do training and validation.
        if len(self.dataloaders) < 2:
            raise ValueError("At least 2 dataloaders required (train and validation)")
        self.training_dataloader = self.dataloaders[0]
        self.validation_dataloader = self.dataloaders[1]

        # Make the optimizers here because we need to save them in the checkpoints.
        self._make_optimizers(self.optimizer_config)

    def _setup_training(self, load_checkpoint: bool):
        """Completes the setup of the trainer.

        Overriding methods should call this base method first.

        Args:
            load_checkpoint: Whether a checkpoint should be loaded.
        """

        # Add to the trainer the hooks to step the schedulers. By default, all
        # schedulers are stepped at each training iteration.
        self._setup_lr_scheduling()

        if load_checkpoint:
            self.checkpoint_handler.load_checkpoint(self._get_checkpoint_state())
            logging.info(f"Restored checkpoint from {self.working_dir}")

        # Force state to avoid error in case we change number of training steps.
        self.trainer.state.epoch_length = self.steps
        # Setting epoch to 0 is necessary because, if the previous run was completed,
        # the current state has epoch=1 so training will not start.
        self.trainer.state.epoch = 0

        # Initial training iteration (maybe after loading checkpoint)
        iter_start = self.trainer.state.iteration
        logging.info(f"Current training iteration: {iter_start}")
        if iter_start >= self.steps:
            logging.warning(
                f"Skipping training: the maximum number of steps is {self.steps} but "
                f"the checkpoint is at {iter_start}>={self.steps} steps."
            )
            self.trainer.terminate()
            raise Exception()

        self.evaluator = MetricsEvaluator(
            dataloader=self.validation_dataloader,
            device=self.device,
            loss_terms=self.loss_terms,
            skip_background=True,
        )

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.logvalid_steps))
        def evaluate(trainer):
            logging.info("Starting evaluation")
            self.model.eval()
            losses, metrics = self.evaluator.eval(self.model)
            print(f"Checkpoint {self.logvalid_steps} steps: validate loss: {losses['loss']}, mse: {metrics['mse']}")
            self.logger.log_dict(
                metrics=losses,
                iteration_num=self.trainer.state.iteration,
                group_name="validation losses",
            )
            self.logger.log_dict(
                metrics=metrics,
                iteration_num=self.trainer.state.iteration,
                group_name="validation metrics",
            )
            self.model.train()
            logging.info("Evaluation ended")

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.checkpoint_steps))
        def save_checkpoint(engine):
            state_dicts = extract_state_dicts(self._get_checkpoint_state())
            self.checkpoint_handler.save_checkpoint(state_dicts)

        if self.resubmit_steps is not None:
            logging.info(f"Will stop and resubmit every {self.resubmit_steps} steps")

            @self.trainer.on(Events.ITERATION_COMPLETED(every=self.resubmit_steps))
            def stop_training_resubmit_steps(engine):
                if engine.state.iteration >= self.steps:
                    logging.info(
                        f"Current step {engine.state.iteration} is >= total training "
                        f"steps: training will terminate normally."
                    )
                    engine.terminate()
                    return
                logging.info(
                    f"Training ended at iteration {engine.state.iteration}: automatic resubmission "
                    f"every {self.resubmit_steps} iterations. Will exit with exit code 3."
                )
                engine.terminate()
                raise Exception()

        if self.resubmit_hours is not None:
            logging.info(f"Will stop and resubmit every {self.resubmit_hours} hours")

            @self.trainer.on(
                Events.ITERATION_COMPLETED(every=self.checkpoint_steps)
            )  # approximately
            def stop_training_resubmit_hours(engine):
                diff = (time.perf_counter() - self.training_start) / 3600
                if diff < self.resubmit_hours:
                    return
                if engine.state.iteration >= self.steps:
                    logging.info(
                        f"Current step {engine.state.iteration} is >= total training "
                        f"steps: training will terminate normally."
                    )
                    engine.terminate()
                    return
                logging.info(
                    f"Training ended at iteration {engine.state.iteration}: automatic resubmission "
                    f"at the first checkpointing event after {self.resubmit_hours} hours (now at "
                    f"{diff} hours). Will exit with exit code 3."
                )
                engine.terminate()
                raise Exception()

    def train(self):
        self.training_start = time.perf_counter()
        self.trainer.run(
            self.training_dataloader, max_epochs=1, epoch_length=self.steps
        )

    def _get_checkpoint_state(self) -> dict:
        state = dict(model=self.model, trainer=self.trainer)
        state.update({f"opt_{i}": opt for i, opt in enumerate(self.optimizers)})
        # LR schedulers are not necessarily present
        if hasattr(self, "lr_schedulers"):
            state.update(
                {
                    f"lr_scheduler_{i}": scheduler
                    for i, scheduler in enumerate(self.lr_schedulers)
                }
            )
        logging.debug(f"State keys: {list(state.keys())}")
        return state


def ari_(
        true_mask: Tensor, pred_mask: Tensor, num_ignored_objects: int
) -> torch.FloatTensor:
    """Computes the ARI score.

    Args:
        true_mask: tensor of shape [batch_size x *] where values go from 0 to the number of objects.
        pred_mask:  tensor of shape [batch_size x *] where values go from 0 to the number of objects.
        num_ignored_objects: number of objects (in ground-truth mask) to be ignored when computing ARI.

    Returns:
        a vector of ARI scores, of shape [batch_size, ].
    """
    true_mask = true_mask.flatten(1)
    pred_mask = pred_mask.flatten(1)
    not_bg = true_mask >= num_ignored_objects
    result = []
    batch_size = len(true_mask)
    for i in range(batch_size):
        ari_value = adjusted_rand_score(
            true_mask[i][not_bg[i]], pred_mask[i][not_bg[i]]
        )
        result.append(ari_value)
    result = torch.FloatTensor(result)  # shape (batch_size, )
    return result


def compute_iou(mask1: Tensor, mask2: Tensor) -> Tensor:
    intersection = (mask1 * mask2).sum((1, 2, 3))
    union = (mask1 + mask2).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(
        union == 0, torch.tensor(-100.0), intersection.float() / union.float()
    )


def segmentation_covering(
        true_mask: Tensor, pred_mask: Tensor, num_ignored_objects: int
) -> Tuple[Tensor, Tensor]:
    """Returns the segmentation covering of the ground-truth masks by the predicted masks.

    Args:
        true_mask: Ground-truth object masks.
        pred_mask: Predicted object masks.
        num_ignored_objects: The first `num_ignored_objects` objects in the
            ground-truth masks are ignored. Assuming the first objects are
            background objects, this can be used to compute the covering of
            the _foreground_ objects only.

    Returns:
        A tuple containing the Segmentation Covering score (SC) and the Mean
        Segmentation Covering score (mSC).
    """

    assert true_mask.shape == pred_mask.shape, f"{true_mask.shape} - {pred_mask.shape}"
    assert true_mask.shape[1] == 1 and pred_mask.shape[1] == 1
    assert true_mask.min() >= 0
    assert pred_mask.min() >= 0
    bs = true_mask.shape[0]

    n = torch.tensor(bs * [0])
    mean_scores = torch.tensor(bs * [0.0])
    scaling_sum = torch.tensor(bs * [0])
    scaled_scores = torch.tensor(bs * [0.0])

    # Remove ignored objects.
    true_mask_filtered = true_mask[true_mask >= num_ignored_objects]

    # Unique label indices
    labels_true_mask = torch.unique(true_mask_filtered).tolist()
    labels_pred_mask = torch.unique(pred_mask).tolist()

    for i in labels_true_mask:
        true_mask_i = true_mask == i
        if not true_mask_i.any():
            continue
        max_iou = torch.tensor(bs * [0.0])

        # Loop over labels_pred_mask to find max IOU
        for j in labels_pred_mask:
            pred_mask_j = pred_mask == j
            if not pred_mask_j.any():
                continue
            iou = compute_iou(true_mask_i, pred_mask_j)
            max_iou = torch.where(iou > max_iou, iou, max_iou)

        n = torch.where(true_mask_i.sum((1, 2, 3)) > 0, n + 1, n)
        mean_scores += max_iou
        scaling_sum += true_mask_i.sum((1, 2, 3))
        scaled_scores += true_mask_i.sum((1, 2, 3)).float() * max_iou

    mean_sc = mean_scores / torch.max(n, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    return mean_sc, scaled_sc
