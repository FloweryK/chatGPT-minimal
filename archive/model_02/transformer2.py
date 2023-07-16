import math
import torch
import torch.nn as nn


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):
    def __init__(self, config, dropout=0.0):
        super().__init__()
        self.config = config

        self.transformer = nn.Transformer(
            config.d_emb, 
            config.n_head, 
            dim_feedforward=config.d_hidden, 
            num_encoder_layers=config.n_layer, 
            num_decoder_layers=config.n_layer,
            dropout=dropout
        )
        
        self.inputEmbedding = InputEmbedding(config)
        self.positionalEmbedding = PositionalEmbedding(config)
    
    def forward(self, x_enc, x_dec):
        # mask
        enc_mask = generate_square_subsequent_mask(x_enc.size(1)).to(x_enc.device)
        enc_padding_mask = x_enc.eq(0)
        dec_mask = generate_square_subsequent_mask(x_dec.size(1)).to(x_dec.device)
        dec_padding_mask = x_dec.eq(0)

        # mask embedding
        x_enc = self.inputEmbedding(x_enc) + self.positionalEmbedding(x_enc)
        x_dec = self.inputEmbedding(x_dec) + self.positionalEmbedding(x_dec)

        output = self.transformer(
            x_enc.transpose(0, 1), 
            x_dec.transpose(0, 1), 
            enc_mask, 
            dec_mask, 
            src_key_padding_mask=enc_padding_mask, 
            tgt_key_padding_mask=dec_padding_mask
        )
        output = output.transpose(0, 1)

        return output