import torch
import torch.nn as nn
from transformer import Transformer
torch.set_printoptions(linewidth=10000, edgeitems=3)


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = Transformer(config)
        self.linear = nn.Linear(config.d_emb, config.n_vocab)

    def forward(self, x_enc, x_dec):
        # transformer
        y_dec = self.transformer(x_enc, x_dec)

        # linear: (n_batch, n_seq_dec, n_vocab)
        y = self.linear(y_dec)
        y = y.transpose(1, 2)

        return y
