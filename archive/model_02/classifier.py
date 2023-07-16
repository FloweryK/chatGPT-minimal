import torch.nn as nn
from archive.model_02.transformer import Transformer
import config


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = Transformer()
        self.linear = nn.Linear(config.d_emb, config.n_vocab)
    
    def forward(self, x_enc, x_dec):
        # x_enc: (n_batch, n_seq_enc)
        # x_dec: (n_batch, n_seq_dec)

        # x: (n_batch, n_seq_dec, n_emb)
        x = self.transformer(x_enc, x_dec)

        # x: (n_batch, n_seq_dec, n_vocab)
        x = self.linear(x)
        x = x.transpose(1, 2)
        return x

        