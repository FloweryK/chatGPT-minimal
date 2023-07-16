import torch
import torch.nn as nn
from model.embedding import InputEmbedding, PositionalEmbedding
from model.attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = MultiHeadAttention(config)
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
    def __init__(self, config):
        super().__init__()

        self.inputEmbedding = InputEmbedding(config)
        self.positionalEmbedding = PositionalEmbedding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
    
    def forward(self, x):
        # x: (n_batch, n_seq)

        # mask: (n_batch, n_seq, n_seq)
        mask = x.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)

        # x: (n_batch, n_seq, d_emb)
        x = self.inputEmbedding(x) + self.positionalEmbedding(x)

        # => (n_batch, n_seq, d_emb)
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention_self = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.d_emb)

        self.attention_encoder = MultiHeadAttention(config)
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
    def __init__(self, config):
        super().__init__()

        self.inputEmbedding = InputEmbedding(config)
        self.positionalEmbedding = PositionalEmbedding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
    
    def forward(self, x_dec, x_enc, y_enc):
        # x_dec: (n_batch, n_seq_dec)
        # x_enc: (n_batch, n_seq_enc)
        # y_enc: (n_batch, n_seq_enc, n_emb)
        
        # mask_dec_self: (n_batch, n_seq_dec, n_seq_dec)
        mask_dec_pad = x_dec.eq(0).unsqueeze(1).repeat(1, x_dec.size(1), 1)
        mask_dec_ahead = torch.ones_like(mask_dec_pad).triu(diagonal=1)
        mask_dec_self = torch.gt(mask_dec_pad, mask_dec_ahead)

        # mask_dec_enc: (n_batch, n_seq_dec, n_seq_enc)
        mask_dec_enc = x_enc.eq(0).unsqueeze(1).repeat(1, x_dec.size(1), 1)

        # x_dec: (n_batch, n_seq_dec, d_emb)
        x_dec = self.inputEmbedding(x_dec) + self.positionalEmbedding(x_dec)
        
        # => (n_batch, n_seq_dec, d_emb)
        for layer in self.layers:
            x_dec = layer(x_dec, y_enc, mask_dec_self, mask_dec_enc)
        
        return x_dec