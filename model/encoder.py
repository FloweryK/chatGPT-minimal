import torch.nn as nn
import config
from model.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.attention = MultiHeadAttention()
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_emb)

        self.feedforward = nn.Linear(config.d_emb, config.d_emb)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.d_emb)
    
    def forward(self, x_enc, mask_enc_self):
        y_enc = self.norm1(x_enc + self.dropout1(self.attention(x_enc, x_enc, x_enc, mask_enc_self)))
        y_enc = self.norm2(y_enc + self.dropout2(self.feedforward(y_enc)))
        
        return y_enc


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(config.n_layer)])
    
    def forward(self, x_enc, mask_enc_self):
        # x_enc: (n_batch, n_seq_enc)
        # mask: (n_batch, n_seq_enc, n_seq_enc)

        for layer in self.layers:
            y_enc = layer(x_enc, mask_enc_self)

        return y_enc