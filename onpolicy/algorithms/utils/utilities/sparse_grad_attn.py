'''
Giving an N x M attention matrix, returns the same matrix,
but performs masking to determine where to block gradients.
'''

import numpy as np
import torch
from torch.autograd import Function

from .sparse_attn import Sparse_attention


class blocked_grad(Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        # Ensure mask is on the same device as grad_output
        mask = mask.to(grad_output.device)
        return grad_output * mask, mask * 0.0


class Sparse_grad_attention(Function):

    @staticmethod
    def forward(ctx, inp, sa):
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        # Ensure sparsified is on the same device as grad_output
        sparsified = sparsified.to(grad_output.device)
        return grad_output * (sparsified > 0.0).float()


if __name__ == "__main__":
    k = 2
    sa = Sparse_attention(k)

    x = torch.from_numpy(np.array([[[0.1, 0.0, 0.3, 0.2, 0.4],
                                    [0.5, 0.4, 0.1, 0.0, 0.0]]]))
    x = x.reshape((2, 5)).float()

    x.requires_grad = True

    print(x)
    output = Sparse_grad_attention.apply(x, sa)
    print('output', output)

    output.sum().backward()
    print('sparse grad', x.grad)

    x.grad.zero_()
    output = sa(x)
    output.sum().backward()

    print('normal grad', x.grad)
    