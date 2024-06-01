import torch
from torch import nn
import math
import torch.nn.functional as F
from onpolicy.algorithms.utils.SLOTATT.cov2d_grad_fix import conv2d
from torch.nn.modules.utils import _pair


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2, inplace=True) * 1.4

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
            stride=1,
            padding=1
    ):
        layers = []

        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, stride=1, padding=1)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, stride=1, padding=1)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False, stride=1, padding=0
        )

    def forward(self, input):
        out = self.conv1(input) * 1.4
        out = self.conv2(out) * 1.4

        skip = self.skip(input) * 1.4
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size=64, ndf=64, image_size=44, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: ndf * 2,
            8: ndf * 2,
            16: ndf,
            32: ndf,
            64: ndf // 2,
            128: ndf // 2
        }

        convs = [ConvLayer(3, channels[size], 1, stride=1, padding=1)]  # +2
        final_img_size = conv2d_output_shape(image_size, image_size, 1, 1, 1)[0]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            final_img_size = compute_res_out_shape(final_img_size)

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, stride=1, padding=1)
        final_img_size = conv2d_output_shape(final_img_size, final_img_size, 3, 1, 1)[0]

        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * final_img_size * final_img_size, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input) * 1.4

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out) * 1.4

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


def compute_res_out_shape(input_size):
    out = conv2d_output_shape(input_size, input_size, 3, 1, 1)
    out = conv2d_output_shape(out[0], out[1], 2, 2, 0)
    out = conv2d_output_shape(out[0], out[1], 3, 1, 1)
    return out[0]


def conv2d_output_shape(h_in, w_in, kernel_size, stride=1, padding=0, dilation=1):
    r"""Determine output shape after a 2D convolutional layer.

    :param h_in: Height of input image.
    :param w_in: Width of input image.
    :param kernel_size: Dimensions of the convolutional kernel.
    :param stride: Stride of convolutional operation.
    :param padding: Padding of convolutional operation.
    :param dilation: Dilation of convolutional operation.

    :return: (Height, Width) of output image.
    """

    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    dilation = _pair(dilation)
    stride = _pair(stride)
    h_out = (
                    (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]
            ) + 1
    w_out = (
                    (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]
            ) + 1
    return (int(h_out), int(w_out))
