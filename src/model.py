from torch import tanh

from layers import MaskedAttention


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_size, emb, max_len=64, dropout=0.5) -> None:
        super().__init__()

        self._emb = nn.Embedding(vocab_size, embedding_size, max_len, _weight=emb)
        self.dropout = nn.Dropout(p=dropout)
        self._encoder = nn.LSTM(embedding_size, embedding_size, bidirectional=True)
        self._soft_attention = MaskedAttention()
        self._projector = nn.Linear(embedding_size * 8, embedding_size)
        self._compositor = nn.LSTM(embedding_size, embedding_size, bidirectional=True)
        self._classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(embedding_size * 8, embedding_size),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(embedding_size, 2),
                nn.Softmax(dim=-1)
                )
        self.apply(_init_weights)

    def forward(self, premises, premises_mask, hypotheses, hypotheses_mask):
        '''
        params 
            premises: (S, N, H)
            hypotheses: (T, N, H)
        '''

        logging.debug(f"mask shape: {premises_mask.shape}")

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
        logging.debug(f"encoded: {encoded_premises.shape}")
        logging.debug(f"encoded: {encoded_hypotheses.shape}")

        # (L, N, 2 * E)

        interacted_premises, interacted_hypotheses, _, _ = self._soft_attention(encoded_premises, premises_mask, encoded_hypotheses, hypotheses_mask)
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
        projected_premises = self.dropout(projected_premises)
        projected_hypotheses = self.dropout(projected_hypotheses)
        logging.debug(f"projected: {projected_hypotheses.shape}")

        # (L, N, E)
        
        composited_premises, _ = self._compositor(projected_premises)
        composited_hypotheses, _ = self._compositor(projected_hypotheses)
        logging.debug(f"composited: {composited_hypotheses.shape}")

        # (L, N, 2 * E)

        avg_premises = torch.mean(composited_premises, 0, keepdim=True)
        avg_hypotheses = torch.mean(composited_hypotheses, 0, keepdim=True)
        max_premises, _ = torch.max(composited_premises, 0, keepdim=True)
        max_hypotheses, _ = torch.max(composited_hypotheses, 0, keepdim=True)
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
        
        probs = self._classifier(final_features)

        logging.debug(f"probs: {probs.shape}")
        return {
                "probs": probs,
                "label": torch.argmax(probs, dim=-1)
                }



class ESIMV1(nn.Module):
    def __init__(self, vocab_size, embedding_size, emb, max_len=64, dropout=0.5) -> None:
        super().__init__()

        self._emb = nn.Embedding(vocab_size, embedding_size, max_len, _weight=emb)
        self.dropout = nn.Dropout(p=dropout)
        self._transformer = nn.TransformerEncoderLayer(embedding_size, 8, dim_feedforward= embedding_size * 4, activation="relu")
        self._encoder = nn.TransformerEncoder(self._transformer, 3)
        self._soft_attention = MaskedAttention()
        self._projector = nn.Linear(embedding_size * 4, embedding_size)
        self._compositor = nn.TransformerEncoder(self._transformer, 3)
        self._classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(embedding_size * 4, embedding_size),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(embedding_size, 2),
                nn.Softmax(dim=-1)
                )
        self.apply(_init_weights)

    def forward(self, premises, premises_mask, hypotheses, hypotheses_mask):
        '''
        params 
            premises: (S, N, H)
            hypotheses: (T, N, H)
        '''

        logging.debug(f"mask shape: {premises_mask.shape}")

        premises_mask_bool = premises_mask == 0
        premises_mask_bool = premises_mask_bool.squeeze(1)
        hypotheses_mask_bool = hypotheses_mask == 0
        hypotheses_mask_bool = hypotheses_mask_bool.squeeze(1)

        emb_premises = self._emb(premises)
        emb_hypotheses = self._emb(hypotheses)
        logging.debug(emb_hypotheses.shape)

        # (N, L, E)
        emb_premises = torch.transpose(emb_premises, 0, 1)
        emb_hypotheses = torch.transpose(emb_hypotheses, 0, 1)
        emb_premises = self.dropout(emb_premises)
        emb_hypotheses = self.dropout(emb_hypotheses)
        logging.debug(f"emb: {emb_hypotheses.shape}")
        # (L, N, E)

        encoded_premises = self._encoder(emb_premises, src_key_padding_mask=premises_mask_bool)
        encoded_hypotheses = self._encoder(emb_hypotheses, src_key_padding_mask=hypotheses_mask_bool)
        logging.debug(f"encoded: {encoded_premises.shape}")
        logging.debug(f"encoded: {encoded_hypotheses.shape}")

        # (L, N, 2 * E)

        interacted_premises, interacted_hypotheses, _, _ = self._soft_attention(encoded_premises, premises_mask, encoded_hypotheses, hypotheses_mask)
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
        projected_premises = self.dropout(projected_premises)
        projected_hypotheses = self.dropout(projected_hypotheses)
        logging.debug(f"projected: {projected_hypotheses.shape}")

        # (L, N, E)
        
        composited_premises = self._compositor(projected_premises, src_key_padding_mask=premises_mask_bool)
        composited_hypotheses = self._compositor(projected_hypotheses, src_key_padding_mask=hypotheses_mask_bool)
        logging.debug(f"composited: {composited_hypotheses.shape}")

        # (L, N, 2 * E)

        avg_premises = torch.mean(composited_premises, 0, keepdim=True)
        avg_hypotheses = torch.mean(composited_hypotheses, 0, keepdim=True)
        max_premises, _ = torch.max(composited_premises, 0, keepdim=True)
        max_hypotheses, _ = torch.max(composited_hypotheses, 0, keepdim=True)
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
        
        probs = self._classifier(final_features)

        logging.debug(f"probs: {probs.shape}")
        return {
                "probs": probs,
                "label": torch.argmax(probs, dim=-1)
                }

def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size: (2 * hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size: (2 * hidden_size)] = 1.0


class ESIMV2(nn.Module):
    def __init__(self, vocab_size, embedding_size, emb, max_len=64, dropout=0.5) -> None:
        super().__init__()

        self._emb = nn.Embedding(vocab_size, embedding_size, max_len, _weight=emb)
        self.dropout = nn.Dropout(p=dropout)
        self._transformer = nn.TransformerEncoderLayer(embedding_size, 4, dim_feedforward= embedding_size * 4, activation="gelu")
        self._encoder = nn.TransformerEncoder(self._transformer, 3)
        self._soft_attention = MaskedAttention()
        self._projector = nn.Linear(embedding_size * 4, embedding_size)
        self._compositor = nn.TransformerEncoder(self._transformer, 3)
        self._classifier = nn.Sequential(
                nn.Linear(max_len * 4, 2),
                nn.Softmax(dim=-1)
                )
        self.apply(_init_weights)

    def forward(self, premises, premises_mask, hypotheses, hypotheses_mask):
        '''
        params 
            premises: (S, N, H)
            hypotheses: (T, N, H)
        '''

        logging.debug(f"mask shape: {premises_mask.shape}")

        premises_mask_bool = premises_mask == 0
        premises_mask_bool = premises_mask_bool.squeeze(1)
        hypotheses_mask_bool = hypotheses_mask == 0
        hypotheses_mask_bool = hypotheses_mask_bool.squeeze(1)

        emb_premises = self._emb(premises)
        emb_hypotheses = self._emb(hypotheses)
        logging.debug(emb_hypotheses.shape)

        # (N, L, E)
        emb_premises = torch.transpose(emb_premises, 0, 1)
        emb_hypotheses = torch.transpose(emb_hypotheses, 0, 1)
        emb_premises = self.dropout(emb_premises)
        emb_hypotheses = self.dropout(emb_hypotheses)
        logging.debug(f"emb: {emb_hypotheses.shape}")
        # (L, N, E)

        encoded_premises = self._encoder(emb_premises, src_key_padding_mask=premises_mask_bool)
        encoded_hypotheses = self._encoder(emb_hypotheses, src_key_padding_mask=hypotheses_mask_bool)
        logging.debug(f"encoded: {encoded_premises.shape}")
        logging.debug(f"encoded: {encoded_hypotheses.shape}")

        # (L, N, 2 * E)

        interacted_premises, interacted_hypotheses, _, _ = self._soft_attention(encoded_premises, premises_mask, encoded_hypotheses, hypotheses_mask)
        logging.debug(f"interacted: {interacted_hypotheses.shape}")

        # (L, N, 2 * E)

        interacted_premises = self.dropout(interacted_premises)
        interacted_hypotheses = self.dropout(interacted_hypotheses)
        logging.debug(f"interacted: {interacted_hypotheses.shape}")

        # (L, N, E)
        
        composited_premises = self._compositor(interacted_premises, src_key_padding_mask=premises_mask_bool)
        composited_hypotheses = self._compositor(interacted_hypotheses, src_key_padding_mask=hypotheses_mask_bool)
        logging.debug(f"composited: {composited_hypotheses.shape}")

        # (L, N, 2 * E)

        _, _, score_hypotheses, _ = self._soft_attention(encoded_premises, premises_mask, composited_premises, premises_mask)
        _, _, score_premises, _ = self._soft_attention(encoded_hypotheses, hypotheses_mask, composited_hypotheses, hypotheses_mask)


        final_features = torch.cat((
                score_premises,
                score_hypotheses
            ), -1)

        logging.debug(f"final_features: {final_features.shape}")
        
        avg = torch.mean(final_features, 1, keepdim=True)
        maxi, _ = torch.max(final_features, 1, keepdim=True)
        logging.debug(f"max features: {maxi.shape}")

        avg = torch.squeeze(avg, 1)
        maxi = torch.squeeze(maxi, 1)
        logging.debug(f"max features: {maxi.shape}")
        final_features = torch.cat(
                    (avg, maxi), -1
                )
        probs = self._classifier(final_features)
        logging.debug(f"probs: {probs.shape}")

        return {
                "probs": probs,
                "label": torch.argmax(probs, dim=-1)
                }

def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size: (2 * hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size: (2 * hidden_size)] = 1.0
