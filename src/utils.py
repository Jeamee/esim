import torch.nn.functional as F


def masked_softmax(tensor, mask):
    tensor = F.softmax(tensor, dim=-1)

    mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    tensor = tensor * mask
    tensor = tensor / (tensor.sum(dim=-1, keepdim=True) + 1e-13)
    
    return tensor
