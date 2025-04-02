import os
import sys
from typing import Tuple
from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels: Tuple[int, ...], strides: Tuple[int, ...], kernel_size):
        super().__init__()
        modules = []
        channel = 3
        for ch, s in zip(channels, strides):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channel, ch, kernel_size, stride=s, padding=kernel_size // 2),
                    nn.ReLU(),
                )
            )
            channel = ch
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        """
        input:
            x: input image, [B, 3, H, W]
        output:
            feature_map: [B, C, H_enc, W_enc]
        """
        # Use gradient checkpointing to save memory if available
        if hasattr(torch, 'utils') and hasattr(torch.utils, 'checkpoint'):
            from torch.utils.checkpoint import checkpoint
            
            # Process in chunks to save memory
            modules = list(self.conv.children())
            
            # Apply first convolution
            out = modules[0](x)
            
            # Apply remaining convolutions with checkpointing
            for module in modules[1:]:
                def custom_forward(input_tensor):
                    return module(input_tensor)
                out = checkpoint(custom_forward, out, use_reentrant=False)
            
            return out
        else:
            # Fallback to regular forward if checkpointing is not available
            x = self.conv(x)
            return x



