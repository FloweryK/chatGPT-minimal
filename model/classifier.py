import torch
import torch.nn as nn
import config
from transformer import Transformer
from transformer.embedding import Embedding
torch.set_printoptions(linewidth=10000, edgeitems=3)


class Transformer2(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding()
        self.transformer = nn.Transformer(
            d_model=config.d_emb,
            nhead=config.n_head,
            num_encoder_layers=config.n_layer,
            num_decoder_layers=config.n_layer,
            dim_feedforward=config.d_hidden,
            dropout=config.dropout,
        )

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x_enc, x_dec):
        # mask
        mask_enc = self.generate_square_subsequent_mask(x_enc.size(1)).to(x_enc.device)
        mask_enc_padding = x_enc.eq(0)
        mask_dec = self.generate_square_subsequent_mask(x_dec.size(1)).to(x_dec.device)
        mask_dec_padding = x_dec.eq(0)

        # embedding
        x_enc = self.embedding(x_enc)
        x_dec = self.embedding(x_dec)

        y_dec = self.transformer(
            x_enc.transpose(0, 1),
            x_dec.transpose(0, 1),
            mask_enc,
            mask_dec,
            src_key_padding_mask=mask_enc_padding, 
            tgt_key_padding_mask=mask_dec_padding
        )

        y_dec = y_dec.transpose(0, 1)

        return y_dec


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = Transformer()
        self.linear = nn.Linear(config.d_emb, config.n_vocab)

    def forward(self, x_enc, x_dec):
        # transformer
        y_dec = self.transformer(x_enc, x_dec)

        # linear: (n_batch, n_seq_dec, n_vocab)
        y = self.linear(y_dec)
        y = y.transpose(1, 2)

        return y
