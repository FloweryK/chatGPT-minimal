import torch.nn as nn
from transformer.attention import MultiHeadAttention
from transformer.positionwise_feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention_self = MultiHeadAttention(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_emb)

        self.attention_enc = MultiHeadAttention(config)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_emb)

        self.feedforward = PositionwiseFeedForward(config)
        self.dropout3 = nn.Dropout(config.dropout)
        self.norm3 = nn.LayerNorm(config.d_emb)
    
    def forward(self, x_dec, y_enc, mask_dec_self, mask_dec_enc):
        # x_dec: (n_batch, n_seq_dec, d_emb)
        # y_enc: (n_batch, n_seq_dec, d_emb)
        # mask_dec_self: (n_batch, n_head, n_seq_dec, n_seq_dec)
        # mask_dec_enc: (n_batch, n_head, n_seq_dec, n_seq_enc)

        x_dec = self.norm1(x_dec + self.dropout1(self.attention_self(x_dec, x_dec, x_dec, mask_dec_self)))
        x_dec = self.norm2(x_dec + self.dropout2(self.attention_enc(x_dec, y_enc, y_enc, mask_dec_enc)))
        x_dec = self.norm2(x_dec + self.dropout3(self.feedforward(x_dec)))

        return x_dec


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
    
    def forward(self, x_dec, y_enc, mask_dec_self, mask_dec_enc):
        # x_dec: (n_batch, n_seq_dec, d_emb)
        # y_enc: (n_batch, n_seq_enc, d_emb)
        # mask_dec_self: (n_batch, n_seq_enc, n_seq_enc)
        # mask_dec_enc: (n_batch, n_seq_dec, n_seq_enc)

        for layer in self.layers:
            x_dec = layer(x_dec, y_enc, mask_dec_self, mask_dec_enc)
        
        return x_dec
