import torch
import torch.nn as nn
from archive.model_02.embedding import InputEmbedding, PositionalEmbedding
from archive.model_02.encoder import Encoder, Decoder
from constant import *


def make_mask_pad(q, k):
    mask_q = q.eq(0).unsqueeze(2).repeat(1, 1, k.size(1))
    mask_k = k.eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
    mask = mask_q | mask_k
    return mask


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.inputEmbedding = InputEmbedding()
        self.positionalEmbedding = PositionalEmbedding()

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x_enc, x_dec):
        # mask_enc_self
        # mask_enc_self = x_enc.eq(0).unsqueeze(1).repeat(1, x_enc.size(1), 1)
        mask_enc_self = make_mask_pad(x_enc, x_enc)

        # mask_dec_self
        # mask_dec_pad = x_dec.eq(0).unsqueeze(1).repeat(1, x_dec.size(1), 1)
        mask_dec_pad = make_mask_pad(x_dec, x_dec)
        mask_dec_tri = torch.ones_like(mask_dec_pad).triu(diagonal=1)
        mask_dec_self = mask_dec_pad | mask_dec_tri

        # mask_dec_enc
        # mask_dec_enc = x_enc.eq(0).unsqueeze(1).repeat(1, x_dec.size(1), 1)
        mask_dec_enc = make_mask_pad(x_dec, x_enc)
        
        # embedding
        x_enc = self.inputEmbedding(x_enc) + self.positionalEmbedding(x_enc)
        x_dec = self.inputEmbedding(x_dec) + self.positionalEmbedding(x_dec)

        y_enc = self.encoder(x_enc, mask_enc_self)
        y_dec = self.decoder(x_dec, y_enc, mask_dec_self, mask_dec_enc)

        return y_dec