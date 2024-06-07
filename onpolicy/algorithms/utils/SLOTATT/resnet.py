# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of ResNet V1 in Flax.

"Deep Residual Learning for Image Recognition"
He et al., 2015, [https://arxiv.org/abs/1512.03385]
"""

import functools

from typing import Any, Tuple, Type, List, Optional
import torch
from torch import nn

Conv1x1 = functools.partial(nn.Conv2d, kernel_size=1, bias=False)
Conv3x3 = functools.partial(nn.Conv2d, kernel_size=3, bias=False, padding=1, dilation=1)


class ResNetBlock(nn.Module):
    """ResNet block without bottleneck used in ResNet-18 and ResNet-34."""

    def __init__(self,
                 input_channel: int,
                 filters: int,
                 norm: str,
                 kernel_dilation: Tuple[int, int] = (1, 1),
                 strides: Tuple[int, int] = (1, 1),
                 ):
        super().__init__()
        self.input_channel = input_channel
        self.filters = filters
        self.norm_type = norm
        self.kernel_dilation = kernel_dilation
        self.strides = strides

        if self.norm_type == "batch":
            self.norm = functools.partial(nn.BatchNorm2d, momentum=0.9)
            self.bn1 = self.norm(num_features=self.filters)
            self.bn2 = self.norm(num_features=self.filters)
            self.proj_bn = self.norm(num_features=self.filters)
        elif self.norm_type == "group":
            self.norm = functools.partial(nn.GroupNorm, num_groups=32)
            self.bn1 = self.norm(num_channels=self.filters)
            self.bn2 = self.norm(num_channels=self.filters)
            self.proj_bn = self.norm(num_channels=self.filters)
        else:
            raise ValueError(f"Invalid norm_type: {self.norm_type}")

        self.act_fn = torch.nn.ReLU()
        self.conv1 = Conv3x3(
            in_channels=input_channel,
            out_channels=self.filters,
            stride=self.strides)

        self.conv2 = Conv3x3(in_channels=self.filters, out_channels=self.filters)

        self.proj_conv = Conv1x1(in_channels=input_channel, out_channels=self.filters, stride=self.strides)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        x = self.conv2(x)
        # Initializing the scale to 0 has been common practice since "Fixup
        # Initialization: Residual Learning Without Normalization" Tengyu et al,
        # 2019, [https://openreview.net/forum?id=H1gsz30cKX].
        x = self.bn2(x)

        if residual.shape != x.shape:
            residual = self.proj_conv(residual)
            residual = self.proj_bn(residual)

        x = self.act_fn(residual + x)
        return x


class ResNetStage(nn.Module):
    """ResNet stage consistent of multiple ResNet blocks."""

    def __init__(self,
                 input_channel: int,
                 stage_size: int,
                 filters: int,
                 block_cls: Type[ResNetBlock],
                 norm: str,
                 first_block_strides: Tuple[int, int],
                 ):
        super().__init__()
        self.input_channel = input_channel
        self.stage_size = stage_size
        self.filters = filters
        self.block_cls = block_cls
        self.norm = norm
        self.first_block_strides = first_block_strides

        self.blocks = nn.ModuleList()
        for i in range(self.stage_size):
            block = self.block_cls(
                input_channel=input_channel if i == 0 else self.filters,
                filters=self.filters,
                norm=self.norm,
                strides=self.first_block_strides if i == 0 else (1, 1))
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ResNet(nn.Module):
    """Construct ResNet V1 with `num_classes` outputs.

    Attributes:
      num_classes: Number of nodes in the final layer.
      block_cls: Class for the blocks. ResNet-50 and larger use
        `BottleneckResNetBlock` (convolutions: 1x1, 3x3, 1x1), ResNet-18 and
          ResNet-34 use `ResNetBlock` without bottleneck (two 3x3 convolutions).
      stage_sizes: List with the number of ResNet blocks in each stage. Number of
        stages can be varied.
      norm_type: Which type of normalization layer to apply. Options are:
        "batch": BatchNorm, "group": GroupNorm, "layer": LayerNorm. Defaults to
        BatchNorm.
      width_factor: Factor applied to the number of filters. The 64 * width_factor
        is the number of filters in the first stage, every consecutive stage
        doubles the number of filters.
      small_inputs: Bool, if True, ignore strides and skip max pooling in the root
        block and use smaller filter size.
      stage_strides: Stride per stage. This overrides all other arguments.
      include_top: Whether to include the fully-connected layer at the top
        of the network.
    """

    def __init__(self,
                 input_channel: int,
                 block_cls: Type[ResNetBlock],
                 stage_sizes: List[int],
                 norm_type: str = "batch",
                 width_factor: int = 1,
                 small_inputs: bool = False,
                 stage_strides: Optional[List[Tuple[int, int]]] = None,
                 output_initializer=nn.init.zeros_  # some initializer return value, some do not
                 ):
        super().__init__()
        self.input_channel = input_channel
        self.block_cls = block_cls
        self.stage_sizes = stage_sizes
        self.norm_type = norm_type
        self.width_factor = width_factor
        self.small_inputs = small_inputs
        self.stage_strides = stage_strides
        self.output_initializer = output_initializer

        self.width = 64 * self.width_factor
        self.init_conv = nn.Conv2d(
            in_channels=input_channel,
            out_channels=self.width,
            kernel_size=(7, 7) if not self.small_inputs else (3, 3),
            stride=(2, 2) if not self.small_inputs else (1, 1),
            bias=False,
            padding=(3, 3) if not self.small_inputs else (1, 1))

        if self.norm_type == "batch":
            self.init_bn = nn.BatchNorm2d(num_features=self.width, momentum=0.9)
        elif self.norm_type == "group":
            self.init_bn = nn.GroupNorm(num_channels=self.width, num_groups=32)
        else:
            raise ValueError(f"Invalid norm_type: {self.norm_type}")

        self.res_net_stages = nn.ModuleList()
        for i, stage_size in enumerate(self.stage_sizes):
            if i == 0:
                first_block_strides = (
                    1, 1) if self.stage_strides is None else self.stage_strides[i]
            else:
                first_block_strides = (
                    2, 2) if self.stage_strides is None else self.stage_strides[i]

            stage = ResNetStage(
                input_channel=self.width * 2 ** i,
                stage_size=stage_size,
                filters=self.width * 2 ** (i + 1),
                block_cls=self.block_cls,
                norm=self.norm_type,
                first_block_strides=first_block_strides,
            )
            self.res_net_stages.append(stage)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Apply the ResNet to the inputs `x`.

        Args:
          x: Inputs.

        Returns:
          The output head with `num_classes` entries.
        """
        # Root block.
        x = self.init_conv(x)
        x = self.init_bn(x)

        if not self.small_inputs:
            x = torch.max_pool2d(x, (3, 3), stride=(2, 2))  # TODO may need to compute padding manually

        # Stages.
        stage_res = {}
        for idx, stage in enumerate(self.res_net_stages):
            x = stage(x)
            stage_res["C" + str(idx)] = x

        return stage_res


ResNetWithBasicBlk = functools.partial(ResNet, block_cls=ResNetBlock)
ResNet18 = functools.partial(ResNetWithBasicBlk, stage_sizes=[2, 2, 2, 2])
