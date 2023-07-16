import torch.nn as nn
import torch.nn.functional as F
from model.transformer import Transformer


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = Transformer(config)
        self.linear = nn.Linear(config.d_emb, config.n_vocab)
    
    def forward(self, x_enc, x_dec):
        # x_enc: (n_batch, n_seq_enc)
        # x_dec: (n_batch, n_seq_dec)

        # x: (n_batch, n_seq_dec, n_emb)
        x = self.transformer(x_enc, x_dec)

        # x: (n_batch, n_seq_dec, n_vocab)
        x = self.linear(x)
        return x

        