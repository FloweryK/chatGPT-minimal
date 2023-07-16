import torch.nn as nn
import config
from model.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_self = MultiHeadAttention()
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_emb)

        self.attention_enc = MultiHeadAttention()
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_emb)

        self.feedforward = nn.Linear(config.d_emb, config.d_emb)
        self.dropout3 = nn.Dropout(config.dropout)
        self.norm3 = nn.LayerNorm(config.d_emb)
    
    def forward(self, x_dec, y_enc, mask_dec_self, mask_dec_enc):
        y_dec = self.norm1(x_dec + self.dropout1(self.attention_self(x_dec, x_dec, x_dec, mask_dec_self)))
        y_dec = self.norm2(y_dec + self.dropout2(self.attention_enc(y_dec, y_enc, y_enc, mask_dec_enc)))
        y_dec = self.norm2(y_dec + self.dropout3(self.feedforward(y_dec)))

        return y_dec


# COMPLETE
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer() for _ in range(config.n_layer)])
    
    def forward(self, x_dec, y_enc, mask_dec_self, mask_dec_enc):
        # x_dec: (n_batch, n_seq_dec, d_emb)
        # y_enc: (n_batch, n_seq_enc, d_emb)
        # mask_dec_self: (n_batch, n_seq_enc, n_seq_enc)
        # mask_dec_enc: (n_batch, n_seq_dec, n_seq_enc)

        for layer in self.layers:
            y_dec = layer(x_dec, y_enc, mask_dec_self, mask_dec_enc)
        
        return y_dec
