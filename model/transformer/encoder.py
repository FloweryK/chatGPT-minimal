import torch.nn as nn
from .attention import MultiHeadAttention
from .positionwise_feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = MultiHeadAttention(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_emb)

        self.feedforward = PositionwiseFeedForward(config)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_emb)
    
    def forward(self, x_enc, mask_enc_self):
        # x_enc: (n_batch, n_seq_enc, d_emb)
        # mask_enc_self: (n_batch, n_head, n_seq_enc, n_seq_enc)

        x_enc = self.norm1(x_enc + self.dropout1(self.attention(x_enc, x_enc, x_enc, mask_enc_self)))
        x_enc = self.norm2(x_enc + self.dropout2(self.feedforward(x_enc)))
        
        return x_enc


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
    
    def forward(self, x_enc, mask_enc_self):
        # x_enc: (n_batch, n_seq_enc, d_emb)
        # mask_enc_self: (n_batch, n_head, n_seq_enc, n_seq_enc)

        for layer in self.layers:
            x_enc = layer(x_enc, mask_enc_self)

        return x_enc