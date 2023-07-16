import torch.nn as nn
from model.encoder import Encoder, Decoder


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, x_enc, x_dec):
        y_enc = self.encoder(x_enc)
        y_dec = self.decoder(x_dec, x_enc, y_enc)

        return y_dec