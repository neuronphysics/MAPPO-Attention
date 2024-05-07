import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=[32, 64],
        kernel_sizes=[4, 3],
        strides=[2, 2],
        hidden_layer=512,
        out_size=64,):
        
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.hidden_layer = hidden_layer
        self.out_size = out_size
        

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_sizes[0], strides[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_sizes[1], strides[1])
        self.linear1 = nn.Linear(64, hidden_layer)
        self.linear11 = nn.Linear(2304, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, out_size)

    def forward(self, inputs):
        #torch.Size([9, 3, 28, 28]) <class 'torch.Tensor'> inputs convolution, error here , marlgrid
        #                3 32 4 2

        #print(inputs.shape, type(inputs), "inputs convolution, error here")
        #print(self.in_channels, self.channels[0], self.kernel_sizes[0], self.strides[0])
        #meltingpot,     torch.Size([9009, 3, 11, 11]) <class 'torch.Tensor'> inputs convolution, error here
        #                3 32 4 2
        x = F.relu(self.conv1(inputs / 255.))
        x = F.relu(self.conv2(x))
        #print(x.shape, "x shape")
        if x.size(0) > 9:
            x = x.view(-1, 64)  # Reshape x to have a first dimension of size 64 if its current first dimension is greater than 9
            #x=x.view(-1, 64 * 6 * 6)
            x = F.relu(self.linear1(x))
        else:
            x = x.view(-1, 64 * 6 * 6)
            x = F.relu(self.linear11(x))
            
        x = self.linear2(x)

        return x
