import torch.nn as nn
from .transformer import Transformer


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = Transformer(config)
        self.linear = nn.Linear(config.d_emb, config.n_vocab)

    def forward(self, x_enc, x_dec):
        # x_enc: (n_batch, n_seq_enc)
        # x_dec: (n_batch, n_seq_dec)

        # transformer
        # y_dec: (n_batch, n_seq_dec, d_emb)
        y_dec = self.transformer(x_enc, x_dec)

        # linear: (n_batch, n_seq_dec, n_vocab)
        y = self.linear(y_dec)

        # transpose for criterion
        # y: (n_batch, d_emb, n_seq_dec)
        y = y.transpose(1, 2)

        return y
