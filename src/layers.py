from utils import masked_softmax, weighted_sum
from torch.autograd import Variable


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, premises, premises_mask, hypotheses, hypotheses_mask):
        """
        params
            premises: (S, N, H)
            hypotheses: (T, N, H)
            premises mask: (N, S)
            hypotheses maks: (N, T) 
        return
            new_premises: (S, N, H)
            new_hypotheses: (T, N, H)
        """
        premises = premises.transpose(0, 1)
        logging.debug(f"premises shape: {premises.shape}")
        # (N, S, H)
        hypotheses = hypotheses.transpose(0, 1)
        logging.debug(f"hypotheses shape: {hypotheses.shape}")
        # (N, T, H)


        attn_premises = torch.bmm(premises, hypotheses.transpose(1, 2))
        # (N, S, T)
        attn_hypotheses = attn_premises.transpose(1, 2)
        # (N, T, S)

        attn_premises = masked_softmax(attn_premises, premises_mask, hypotheses_mask)
        # (N, S, T)
        attn_hypotheses = masked_softmax(attn_hypotheses, hypotheses_mask, premises_mask)
        # (N, T, S)

        logging.debug(f"weight: {attn_premises.shape}, tensor: {hypotheses.shape}")
        new_premises = weighted_sum(attn_premises, hypotheses)
        # (N, S, H)
        new_hypotheses = weighted_sum(attn_hypotheses, premises)
        # (N, T, H)

        new_premises = new_premises.transpose(0, 1)
        # (S, N, H)
        new_hypotheses = new_hypotheses.transpose(0, 1)
        # (T, N, H)

        return new_premises, new_hypotheses


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
