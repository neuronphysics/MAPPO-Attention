import copy
import numpy as np

import torch
import torch.nn as nn

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
    elif height > 50 or width > 50:
        kernel_size = 4
    else:
        kernel_size = 3
   
    stride = 1  # To keep spatial dimensions same, stride should be 1
   
    # Padding is calculated based on the formula to preserve spatial dimensions:
    # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
    # Since we want output_size to be same as input_size (for stride=1), padding is:
    padding = (kernel_size - 1) // 2
   
    return kernel_size, stride, padding

