from torch import tanh
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_size, emb) -> None:
        super().__init__()

        self._emb = nn.Embedding(vocab_size, embedding_size)
        self._emb.from_pretrained(emb)
        self._encoder = nn.LSTM(embedding_size, embedding_size, bidirectional=True)
        self._soft_attention = nn.MultiheadAttention(embedding_size * 2, 6, dropout=0.1)
        self._projector = nn.Linear(embedding_size * 8, embedding_size)
        self._compositor = nn.LSTM(embedding_size, embedding_size, bidirectional=True)
        self._mlp_1 = nn.Linear(embedding_size * 8, embedding_size)
        self._mlp_2 = nn.Linear(embedding_size, 2)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, premises, hypotheses):
        '''input: premises and  hypotheses shape is (L, N)
        '''

        emb_premises = self._emb(premises)
        emb_hypotheses = self._emb(hypotheses)
        logging.debug(emb_hypotheses.shape)

        # (N, L, E)
        emb_premises = torch.transpose(emb_premises, 0, 1)
        emb_hypotheses = torch.transpose(emb_hypotheses, 0, 1)
        logging.debug(f"emb: {emb_hypotheses.shape}")
        # (L, N, E)

        encoded_premises, _ = self._encoder(emb_premises)
        encoded_hypotheses, _ = self._encoder(emb_hypotheses)
        logging.debug(f"encoded: {encoded_hypotheses.shape}")

        # (L, N, 2 * E)

        interacted_premises, _ = self._soft_attention(encoded_premises, encoded_hypotheses, encoded_hypotheses)
        interacted_hypotheses, _ = self._soft_attention(encoded_hypotheses, encoded_premises, encoded_premises)
        logging.debug(f"interacted: {interacted_hypotheses.shape}")

        # (L, N, 2 * E)

        enhanced_premises = torch.cat(
                    (encoded_premises,
                    interacted_premises,
                    encoded_premises - interacted_premises,
                    torch.mul(encoded_premises ,interacted_premises
                )), 2)
        enhanced_hypotheses = torch.cat(
                    (encoded_hypotheses,
                    interacted_hypotheses,
                    encoded_hypotheses - interacted_hypotheses,
                    torch.mul(encoded_hypotheses ,interacted_hypotheses
                )), 2)
        logging.debug(f"enhanced: {enhanced_hypotheses.shape}")
        # (N, L, 8 * E)

        projected_premises = F.relu(self._projector(enhanced_premises))
        projected_hypotheses = F.relu(self._projector(enhanced_hypotheses))
        logging.debug(f"projected: {projected_hypotheses.shape}")

        # (L, N, E)
        
        composited_premises, _ = self._compositor(projected_premises)
        composited_hypotheses, _ = self._compositor(projected_hypotheses)
        logging.debug(f"composited: {composited_hypotheses.shape}")

        # (L, N, 2 * E)



        avg_premises = torch.mean(composited_premises, 0)
        avg_hypotheses = torch.mean(composited_hypotheses, 0)
        max_premises, _ = torch.max(composited_premises, 0)
        max_hypotheses, _ = torch.max(composited_hypotheses, 0)
        logging.debug(f"max features: {max_premises.shape}")

        avg_premises = torch.squeeze(avg_premises, 0)
        avg_hypotheses = torch.squeeze(avg_hypotheses, 0)
        max_premises = torch.squeeze(max_premises, 0)
        max_hypotheses = torch.squeeze(max_hypotheses, 0)
        logging.debug(f"max features: {max_premises.shape}")

        final_features = torch.cat(
                    (avg_premises,
                    avg_hypotheses,
                    max_premises,
                    max_hypotheses
                ), -1)
        logging.debug(f"final_features: {final_features.shape}")
        
        logits = tanh(self._mlp_1(final_features))
        logging.debug(f"mlp1: {logits.shape}")
        logits = self._mlp_2(logits)
        logging.debug(f"mlp2: {logits.shape}")

        probs = self._softmax(logits)

        return probs


class ESIMV1(nn.Module):
    def __init__(self, vocab_size, embedding_size, emb, max_len=64) -> None:
        super().__init__()

        self._emb = nn.Embedding(vocab_size, embedding_size, max_len)
        self._emb.from_pretrained(emb)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_size, nhead=6, dim_feedforward= embedding_size * 4, dropout=0.2)
        self._encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self._soft_attention = nn.MultiheadAttention(embedding_size, 1, dropout=0.2)
        self._projector = nn.Linear(embedding_size * 4, embedding_size)
        self._compositor = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self._dropout = nn.Dropout (p=0.2)
        self._mlp_1 = nn.Linear(embedding_size * 4, embedding_size)
        self._mlp_2 = nn.Linear(embedding_size, 2)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, premises, premises_mask, hypotheses, hypotheses_mask):
        '''input: premises and  hypotheses shape is (L, N)
        '''

        emb_premises = self._emb(premises)
        emb_hypotheses = self._emb(hypotheses)
        emb_premises += premises_mask
        emb_hypotheses += hypotheses_mask
        logging.debug(emb_hypotheses.shape)

        # (N, L, E)
        emb_premises = torch.transpose(emb_premises, 0, 1)
        emb_hypotheses = torch.transpose(emb_hypotheses, 0, 1)
        logging.debug(f"emb: {emb_hypotheses.shape}")
        # (L, N, E)

        encoded_premises = self._encoder(emb_premises)
        encoded_hypotheses = self._encoder(emb_hypotheses)
        logging.debug(f"encoded: {encoded_hypotheses.shape}")

        # (L, N, E)

        interacted_premises, _ = self._soft_attention(encoded_premises, encoded_hypotheses, encoded_hypotheses)
        interacted_hypotheses, _ = self._soft_attention(encoded_hypotheses, encoded_premises, encoded_premises)
        logging.debug(f"interacted: {interacted_hypotheses.shape}")

        # (L, N, E)

        enhanced_premises = torch.cat(
                    (encoded_premises,
                    interacted_premises,
                    encoded_premises - interacted_premises,
                    torch.mul(encoded_premises ,interacted_premises
                )), 2)
        enhanced_hypotheses = torch.cat(
                    (encoded_hypotheses,
                    interacted_hypotheses,
                    encoded_hypotheses - interacted_hypotheses,
                    torch.mul(encoded_hypotheses ,interacted_hypotheses
                )), 2)
        logging.debug(f"enhanced: {enhanced_hypotheses.shape}")
        # (N, L, 4 * E)

        projected_premises = F.relu(self._projector(enhanced_premises))
        projected_hypotheses = F.relu(self._projector(enhanced_hypotheses))
        logging.debug(f"projected: {projected_hypotheses.shape}")

        # (L, N, E)
        
        composited_premises = self._compositor(projected_premises)
        composited_hypotheses = self._compositor(projected_hypotheses)
        logging.debug(f"composited: {composited_hypotheses.shape}")

        # (L, N, E)



        avg_premises = torch.mean(composited_premises, 0)
        avg_hypotheses = torch.mean(composited_hypotheses, 0)
        max_premises, _ = torch.max(composited_premises, 0)
        max_hypotheses, _ = torch.max(composited_hypotheses, 0)
        logging.debug(f"max features: {max_premises.shape}")

        avg_premises = torch.squeeze(avg_premises, 0)
        avg_hypotheses = torch.squeeze(avg_hypotheses, 0)
        max_premises = torch.squeeze(max_premises, 0)
        max_hypotheses = torch.squeeze(max_hypotheses, 0)
        logging.debug(f"max features: {max_premises.shape}")

        final_features = torch.cat(
                    (avg_premises,
                    avg_hypotheses,
                    max_premises,
                    max_hypotheses
                ), -1)
        logging.debug(f"final_features: {final_features.shape}")
        
        logits = tanh(self._mlp_1(final_features))
        logits = self._dropout(logits)
        logging.debug(f"mlp1: {logits.shape}")
        logits = self._mlp_2(logits)
        logging.debug(f"mlp2: {logits.shape}")

        probs = self._softmax(logits)

        return probs
