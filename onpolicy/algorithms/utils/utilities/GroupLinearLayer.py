
import torch
import torch.nn as nn
import math


class GroupLinearLayer(nn.Module):
    def __init__(self, din, dout, num_blocks, bias=True, a=None, device='cpu'):
        super(GroupLinearLayer, self).__init__()
        self.nb = num_blocks
        self.device = device  # Make sure device is defined
        self.dout = dout
        if a is None:
            a = 1. / math.sqrt(dout)
        
        # Move weight to the correct device
        self.weight = nn.Parameter(torch.FloatTensor(num_blocks,din,dout).uniform_(-a,a)).to(self.device)
        
        self.bias = bias
        if bias is True:
            # Move bias to the correct device
            self.bias = nn.Parameter(torch.FloatTensor(num_blocks,dout).uniform_(-a,a)).to(self.device)
        else:
            self.bias = None

    def forward(self,x):
        # Move x to the correct device
        x = x.to(self.device)
        
        ts,bs,m = x.shape
        x = x.permute(1,0,2)
        
        # Now both tensors should be on the same device and this operation should work.
        x = torch.bmm(x,self.weight)
        
        x = x.permute(1,0,2)
        if not self.bias is None:
            x = x + self.bias
        
        return x
