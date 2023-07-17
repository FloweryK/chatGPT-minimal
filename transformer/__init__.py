import torch
import torch.nn as nn
import config
from transformer.embedding import Embedding
from transformer.encoder import Encoder
from transformer.decoder import Decoder


def mask_pad(q, k, n_head):
    mask = k.eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
    mask = mask.unsqueeze(1).repeat(1, n_head, 1, 1)
    return mask


def mask_subsequent(q, k, n_head):
    mask_pad = k.eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
    mask_tri = torch.ones_like(mask_pad).triu(diagonal=1)
    mask = mask_pad | mask_tri
    mask = mask.unsqueeze(1).repeat(1, n_head, 1, 1)
    return mask


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x_enc, x_dec):
        # mask_enc_self: (n_batch, n_head, n_seq_enc, n_seq_enc)
        mask_enc_self = mask_pad(x_enc, x_enc, config.n_head)

        # mask_dec_self: (n_batch, n_head, n_seq_dec, n_seq_dec)
        mask_dec_self = mask_subsequent(x_dec, x_dec, config.n_head)

        # mask_dec_enc: (n_batch, n_head, n_seq_dec, n_seq_enc)
        mask_dec_enc = mask_pad(x_dec, x_enc, config.n_head)

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