import torch
import torch.nn as nn
import config
from model.embedding import Embedding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x_enc, x_dec):
        # mask_enc_self: (n_batch, n_head, n_seq_enc, n_seq_enc)
        mask_enc_self = x_enc.eq(0).unsqueeze(1).repeat(1, x_enc.size(1), 1)
        mask_enc_self = mask_enc_self.unsqueeze(1).repeat(1, config.n_head, 1, 1)

        # mask_dec_self: (n_batch, n_head, n_seq_dec, n_seq_dec)
        mask_dec_pad = x_dec.eq(0).unsqueeze(1).repeat(1, x_dec.size(1), 1)
        mask_dec_tri = torch.ones_like(mask_dec_pad).triu(diagonal=1)
        mask_dec_self = mask_dec_pad | mask_dec_tri
        mask_dec_self = mask_dec_self.unsqueeze(1).repeat(1, config.n_head, 1, 1)

        # mask_dec_enc: (n_batch, n_head, n_seq_dec, n_seq_enc)
        mask_dec_enc = x_enc.eq(0).unsqueeze(1).repeat(1, x_dec.size(1), 1)
        mask_dec_enc = mask_dec_enc.unsqueeze(1).repeat(1, config.n_head, 1, 1)

        # embedding
        # x_enc: (n_batch, n_seq_enc, d_emb)
        # x_dec: (n_batch, n_seq_dec, d_emb)
        x_enc = self.embedding(x_enc)
        x_dec = self.embedding(x_dec)

        # encode & decode
        # y_enc: (n_batch, n_seq_enc, d_emb)
        # y_dec: (n_batch, n_seq_dec, d_emb)
        y_enc = self.encoder(x_enc, mask_enc_self)
        y_dec = self.decoder(x_dec, y_enc, mask_dec_self, mask_dec_enc)

        return y_dec


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