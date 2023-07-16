import torch.nn as nn
from archive.model_02.attention import MultiHeadAttention
import config


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(config.d_emb)
        self.feedforward = nn.Linear(config.d_emb, config.d_emb)
        self.norm2 = nn.LayerNorm(config.d_emb)
    
    def forward(self, x, mask):
        # x: (n_batch, n_seq, d_emb)
        # mask: (n_batch, n_seq, n_seq)
        
        # x: (n_batch, n_seq, d_emb)
        x = self.norm1(x + self.attention(x, x, x, mask))
        x = self.norm2(x + self.feedforward(x))

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(config.n_layer)])
    
    def forward(self, x, mask):
        # x: (n_batch, n_seq, d_emb)
        # => (n_batch, n_seq, d_emb)
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_self = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(config.d_emb)

        self.attention_encoder = MultiHeadAttention()
        self.norm2 = nn.LayerNorm(config.d_emb)

        self.feedforward = nn.Linear(config.d_emb, config.d_emb)
        self.norm3 = nn.LayerNorm(config.d_emb)
    
    def forward(self, x_dec, y_enc, mask_dec_self, mask_dec_enc):
        # x: (n_batch, n_seq, d_emb)
        # mask: (n_batch, n_seq, n_seq)
        
        # x: (n_batch, n_seq, d_emb)
        x_dec = self.norm1(x_dec + self.attention_self(x_dec, x_dec, x_dec, mask_dec_self))
        x_dec = self.norm2(x_dec + self.attention_encoder(x_dec, y_enc, y_enc, mask_dec_enc))
        x_dec = self.norm3(x_dec + self.feedforward(x_dec))

        return x_dec


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer() for _ in range(config.n_layer)])
    
    def forward(self, x_dec, y_enc, mask_dec_self, mask_dec_enc):
        # x_dec: (n_batch, n_seq_dec, n_emb)
        # x_enc: (n_batch, n_seq_enc, n_emb)
        # y_enc: (n_batch, n_seq_enc, n_emb)
        
        # => (n_batch, n_seq_dec, d_emb)
        for layer in self.layers:
            x_dec = layer(x_dec, y_enc, mask_dec_self, mask_dec_enc)
        
        return x_dec