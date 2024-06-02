from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Literal, Optional, Tuple, Union
from onpolicy.algorithms.utils.SLOTATT.discriminator import Discriminator
import numpy as np
from torch.nn import functional as F
import torch
from torch import Tensor, nn
from typing import Callable
from .utils import BaseModel, get_conv_output_shape, make_sequential_from_config, PositionalEmbedding
from torch.optim import Optimizer


class EncoderConfig(Dict):
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    width: int
    height: int
    input_channels: int = 3


class DecoderConfig(Dict):
    conv_tranposes: List[bool]
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    width: int
    height: int
    input_channels: int = 3


class Encoder(nn.Module):
    def __init__(
            self,
            width: int,
            height: int,
            channels: List[int] = (32, 32, 32, 32),
            kernels: List[int] = (5, 5, 5, 5),
            strides: List[int] = (1, 1, 1, 1),
            paddings: List[int] = (2, 2, 2, 2),
            input_channels: int = 3,
            batchnorms: List[bool] = tuple([False] * 4),
    ):
        super().__init__()
        assert len(kernels) == len(strides) == len(paddings) == len(channels)
        self.conv_bone = make_sequential_from_config(
            input_channels,
            channels,
            kernels,
            batchnorms,
            False,
            paddings,
            strides,
            "relu",
            try_inplace_activation=True,
        )
        output_channels = channels[-1]
        output_width, output_height = get_conv_output_shape(
            width, height, kernels, paddings, strides
        )
        self.pos_embedding = PositionalEmbedding(
            output_width, output_height, output_channels
        )
        self.lnorm = nn.GroupNorm(1, output_channels, affine=True, eps=0.001)
        self.conv_1x1 = [
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
        ]
        self.conv_1x1 = nn.Sequential(*self.conv_1x1)

    def forward(self, x: Tensor) -> Tensor:
        conv_output = self.conv_bone(x)
        conv_output = self.pos_embedding(conv_output)
        conv_output = conv_output.flatten(2, 3)  # bs x c x (w * h)
        conv_output = self.lnorm(conv_output)
        return self.conv_1x1(conv_output)


class Decoder(nn.Module):
    def __init__(
            self,
            input_channels: int,
            width: int,
            height: int,
            channels: List[int] = (32, 32, 32, 4),
            kernels: List[int] = (5, 5, 5, 3),
            strides: List[int] = (1, 1, 1, 1),
            paddings: List[int] = (2, 2, 2, 1),
            output_paddings: List[int] = (0, 0, 0, 0),
            conv_transposes: List[bool] = tuple([False] * 4),
            activations: List[str] = tuple(["relu"] * 4),
    ):
        super().__init__()
        self.conv_bone = []
        assert len(channels) == len(kernels) == len(strides) == len(paddings)
        if conv_transposes:
            assert len(channels) == len(output_paddings)
        self.pos_embedding = PositionalEmbedding(width, height, input_channels)
        self.width = width
        self.height = height

        self.conv_bone = make_sequential_from_config(
            input_channels,
            channels,
            kernels,
            False,
            False,
            paddings,
            strides,
            activations,
            output_paddings,
            conv_transposes,
            try_inplace_activation=True,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.pos_embedding(x)
        output = self.conv_bone(x)
        img, mask = output[:, :3], output[:, -1:]
        return img, mask


class SlotAttentionModule(nn.Module):
    def __init__(self, num_slots, channels_enc, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.empty(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, dim))
        with torch.no_grad():
            limit = sqrt(6.0 / (1 + dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(channels_enc, dim, bias=False)
        self.to_v = nn.Linear(channels_enc, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(channels_enc, eps=0.001)
        self.norm_slots = nn.LayerNorm(dim, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=0.001)
        self.dim = dim

    def forward(self, inputs: Tensor, num_slots: Optional[int] = None):
        b, n, _ = inputs.shape
        if num_slots is None:
            num_slots = self.num_slots

        mu = self.slots_mu.expand(b, num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, num_slots, -1).exp()
        slots = mu + sigma * torch.randn_like(sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, self.dim), slots_prev.reshape(-1, self.dim)
            )

            slots = slots.reshape(b, -1, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn


@dataclass(eq=False, repr=False)
class SlotAttentionAE(BaseModel):
    latent_size: int

    encoder_params: Dict
    decoder_params: Dict
    discrim_params: Dict
    discrim_optim_params: Dict
    discrim_train_iter: int
    weight_gan: float
    lose_fn_type: str
    input_channels: int = 3
    eps: float = 1e-8
    mlp_size: int = 128
    attention_iters: int = 3
    w_broadcast: Union[int, Literal["dataset"]] = "dataset"
    h_broadcast: Union[int, Literal["dataset"]] = "dataset"

    encoder: Encoder = field(init=False)
    decoder: Decoder = field(init=False)
    discriminator: Discriminator = field(init=False)
    dis_optimizer: Optimizer = field(init=False)
    loss_fn: Callable = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        self.discriminator = Discriminator(**self.discrim_params)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), **self.discrim_optim_params)

        if self.lose_fn_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.lose_fn_type == "cosine":
            self.loss_fn = lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=-1).mean()
        elif self.lose_fn_type == "l1":
            self.loss_fn = nn.functional.l1_loss

        if self.w_broadcast == "dataset":
            self.w_broadcast = self.width
        if self.h_broadcast == "dataset":
            self.h_broadcast = self.height
        self.encoder_params.update(
            width=self.width, height=self.height, input_channels=self.input_channels
        )
        self.encoder = Encoder(**self.encoder_params)
        self.slot_attention = SlotAttentionModule(
            self.num_slots,
            self.encoder_params["channels"][-1],
            self.latent_size,
            self.attention_iters,
            self.eps,
            self.mlp_size,
        )
        self.decoder_params.update(
            width=self.w_broadcast,
            height=self.h_broadcast,
            input_channels=self.latent_size,
        )
        self.decoder = Decoder(**self.decoder_params)

    @property
    def slot_size(self) -> int:
        return self.latent_size

    def spatial_broadcast(self, slot: Tensor) -> Tensor:
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, self.w_broadcast, self.h_broadcast)

    def slot_similarity_loss(self, slots):
        """
        Calculate the similarity loss for slots with shape (batch, num_slot, hidden_size).
        """
        batch_size = slots.shape[0]
        # Normalize slot features
        slots = F.normalize(slots, dim=-1)  # Normalize along the hidden_size dimension

        # Randomly permute the slots
        perm = torch.randperm(slots.size(1)).to(slots.device)  # Permute along the num_slot dimension

        # Select a subset of n slots
        selected_slots = slots[:, perm[:self.num_slots], :]  # [batch, n, hidden_size]

        # Compute similarity matrix
        sim_matrix = torch.bmm(selected_slots, selected_slots.transpose(1, 2)) * (
                1 / np.sqrt(slots.size(2)))  # [batch, n, n]

        # Create mask to remove diagonal elements (self-similarity)
        mask = torch.eye(self.num_slots).to(slots.device).repeat(batch_size, 1, 1)  # [1, n, n]

        # Mask out the diagonal elements
        sim_matrix = sim_matrix - mask * sim_matrix

        # Compute similarity loss
        sim_loss = sim_matrix.sum(dim=(1, 2)) / (self.num_slots * (self.num_slots - 1))

        return sim_loss.mean()  # Return the mean similarity loss over the batch

    def forward(self, x: Tensor) -> dict:
        with torch.no_grad():
            x = x * 2.0 - 1.0
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1)
        z, attn = self.slot_attention(encoded)
        bs = z.size(0)
        slots = z.flatten(0, 1)
        slots = self.spatial_broadcast(slots)
        img_slots, masks = self.decoder(slots)
        img_slots = img_slots.view(bs, self.num_slots, 3, self.width, self.height)
        masks = masks.view(bs, self.num_slots, 1, self.width, self.height)
        masks = masks.softmax(dim=1)

        recon_slots_masked = img_slots * masks
        recon_img = recon_slots_masked.sum(dim=1)
        loss = self.loss_fn(x, recon_img)

        if self.lose_fn_type == "mse":
            loss = loss + self.slot_similarity_loss(z)

        d_fake = self.discriminator(recon_img)
        loss = loss + self.weight_gan * g_nonsaturating_loss(d_fake.detach())

        with torch.no_grad():
            recon_slots_output = (img_slots + 1.0) / 2.0
        return {
            "loss": loss,  # scalar
            "mask": masks,  # (B, slots, 1, H, W)
            "slot": recon_slots_output,  # (B, slots, 3, H, W)
            "representation": z,  # (B, slots, latent dim)
            #
            "reconstruction": recon_img,  # (B, 3, H, W)
            "attn": attn
        }

    def forward_dis(self, x):
        device = x.device

        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1)
        z, attn = self.slot_attention(encoded)
        bs = z.size(0)
        slots = z.flatten(0, 1)
        slots = self.spatial_broadcast(slots)
        img_slots, masks = self.decoder(slots)
        img_slots = img_slots.view(bs, self.num_slots, 3, self.width, self.height)
        masks = masks.view(bs, self.num_slots, 1, self.width, self.height)
        masks = masks.softmax(dim=1)

        recon_slots_masked = img_slots * masks
        recon_img = recon_slots_masked.sum(dim=1)

        fake_pred = self.discriminator(recon_img.detach())
        real_pred = self.discriminator(x)

        epsilon = torch.rand(len(x), 1, 1, 1, device=device, requires_grad=True)
        gradient = get_gradient(self.discriminator, x, recon_img.detach(), epsilon)
        gp = gradient_penalty(gradient)
        dis_loss = get_dis_loss(fake_pred, real_pred, gp)

        return dis_loss

    def train_discriminator(self, x):
        for i in range(self.discrim_train_iter):
            self.dis_optimizer.zero_grad()
            loss = self.forward_dis(x)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5, norm_type=2)
            self.dis_optimizer.step()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean(), fake_loss.mean()


def get_gradient(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_images = torch.autograd.Variable(mixed_images, requires_grad=True)

    mixed_scores = crit(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.autograd.Variable(torch.ones_like(mixed_scores), requires_grad=False),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    return gradient


def get_dis_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10):
    dis_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    return dis_loss


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty
