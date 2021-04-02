from torch import Tensor


import torch.nn.functional as F
import torch


def masked_softmax(tensor: Tensor, mask1: Tensor, mask2: Tensor) -> Tensor:
    """
    tensor: (N, S, T)
    mask1: (N, S)
    mask2: (N, T)

    return: (N, S, T)
    """
    tensor = F.softmax(tensor, dim=-1)

    mask1 = mask1.unsqueeze(1)
    """(N, S, 1)"""
    mask1 = mask1.expand_as(tensor).contiguous().float()
    """(N, S, T)"""
    mask2 = mask2.unsqueeze(1)
    """(N, T, 1)"""
    mask2 = mask2.expand((tensor.shape[0], mask1.shape[2], mask1.shape[1]))
    """(N, T, S)"""
    mask2 = mask2.transpose(1, 2)
    """(N, S, T)"""
    tensor = tensor * mask1
    tensor = tensor * mask2
    tensor = tensor / (tensor.sum(dim=-1, keepdim=True) + 1e-13)
    """(N, S, T)"""
    
    return tensor


def weighted_sum(tensor, weights):
    """
    tensor: (N, T, H)
    weights: (N, S, T)
    """
    return torch.bmm(weights, tensor)

