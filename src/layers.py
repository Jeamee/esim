from utils import masked_softmax, weighted_sum


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
            premises: (N, S)
            hypotheses: (N, T)

        return
            new_premises: (S, N, H)
            new_hypotheses: (T, N, H)
        """
        premises = premises.transpose(0, 1)
        hypotheses = hypotheses.transpose(0, 1)

        attn_premises = torch.bmm(premises, hypotheses.transpose(1, 2))
        attn_hypotheses = attn_premises.transpose(1, 2)

        attn_premises = masked_softmax(attn_premises, premises_mask, hypotheses_mask)
        attn_hypotheses = masked_softmax(attn_hypotheses, hypotheses_mask, premises_mask)

        new_premises = weighted_sum(attn_premises, hypotheses)
        new_hypotheses = weighted_sum(attn_hypotheses, premises)

        new_premises = new_premises.transpose(0, 1)
        new_hypotheses = new_hypotheses.transpose(0, 1)

        return new_premises, new_hypotheses

