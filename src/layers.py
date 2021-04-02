from utils import masked_softmax


import torch
import torch.nn as nn
import torch.nn.functional as F


class LackAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, premises, premises_mask, hypotheses, hypotheses_mask):
        premises = premises.transpose(0, 1)
        hypotheses = hypotheses.transpose(0, 1)

        attn_premises = torch.bmm(premises, hypotheses.transpose(1, 2))
        attn_hypotheses = attn_premises.transpose(1, 2)

        attn_premises = masked_softmax(attn_premises, premises_mask)
        attn_hypotheses = masked_softmax(attn_hypotheses, hypotheses_mask)

        new_premises = torch.bmm(attn_premises, hypotheses)
        new_hypotheses = torch.bmm(attn_hypotheses, premises)



        



