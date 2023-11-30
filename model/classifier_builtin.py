import torch
import torch.nn as nn
from utils.constant import *


class PositionalEmbedding(nn.Module):
    def __init__(self, d_emb, n_seq):
        super().__init__()
        
        # indices
        i_emb = torch.arange(d_emb)
        i_seq = torch.arange(n_seq).unsqueeze(1)

        # embedding
        self.embedding = i_seq * torch.pow(10000, -2 * i_emb / d_emb)
        self.embedding[:, 0::2] = torch.sin(self.embedding[:, 0::2])
        self.embedding[:, 1::2] = torch.cos(self.embedding[:, 1::2])
        self.embedding = nn.Embedding.from_pretrained(self.embedding, freeze=True)
    
    def forward(self, x):
        n_seq = x.size(-1)
        i_seq = torch.arange(n_seq).to(x.device)

        # x: (n_batch, n_seq, d_emb)
        x = self.embedding(i_seq)
        return x


class Embedding(nn.Module):
    def __init__(self, n_vocab, d_emb, n_seq, dropout=0.1):
        super().__init__()

        self.input_embedding = nn.Embedding(n_vocab, d_emb)
        self.positional_embedding = PositionalEmbedding(d_emb, n_seq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (n_batch, n_seq)
        
        # x: (n_batch, n_seq, d_emb)
        x = self.input_embedding(x) + self.positional_embedding(x)
        x = self.dropout(x)
        return x


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        # layers
        self.embedding = Embedding(
            n_vocab=config.n_vocab, 
            d_emb=config.d_emb, 
            n_seq=config.n_seq
        )
        self.transformer = nn.Transformer(
            d_model=config.d_emb,
            dim_feedforward=config.d_hidden, 
            num_encoder_layers=config.n_layer,
            num_decoder_layers=config.n_layer,
            nhead=config.n_head, 
            dropout=config.dropout,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=config.d_emb, 
            out_features=config.n_vocab
        )
    
    def forward(self, x_enc, x_dec):
        # x_enc: (n_batch, n_seq_enc)
        # x_dec: (n_batch, n_seq_dec)

        # make masks
        src_key_padding_mask = (x_enc == PAD).to(x_enc.device)
        tgt_key_padding_mask = (x_dec == PAD).to(x_dec.device)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(x_dec.size(1)).to(x_dec.device)
        
        # embed inputs
        x_enc = self.embedding(x_enc)
        x_dec = self.embedding(x_dec)

        # transformer forward
        # y_dec: (n_batch, n_seq_dec, d_emb)
        y_dec = self.transformer(
            src=x_enc,
            tgt=x_dec,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask
        )

        # classifier forward
        # linear: (n_batch, n_seq_dec, n_vocab)
        y = self.linear(y_dec)

        # transpose for criterion
        # y: (n_batch, d_emb, n_seq_dec)
        y = y.transpose(1, 2)

        return y