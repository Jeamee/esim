from utils import masked_softmax, weighted_sum


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

