import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.d_emb)
    
    def forward(self, x):
        # x: (n_batch, n_seq)
        # => (n_batch, n_seq, d_emb)
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        # indices
        i_seq = torch.arange(config.n_seq).unsqueeze(1)
        i_emb = torch.arange(config.d_emb)

        # calculate positional embeddings
        self.embedding = torch.zeros(config.n_seq, config.d_emb)
        self.embedding[:, 0::2] = torch.sin(i_seq * torch.pow(10000, -2 * i_emb[0::2] / config.d_emb))
        self.embedding[:, 1::2] = torch.cos(i_seq * torch.pow(10000, -2 * i_emb[1::2] / config.d_emb))

        # make an embedding layer
        self.embedding = nn.Embedding.from_pretrained(self.embedding, freeze=True)
    
    def forward(self, x):
        # x: (n_batch, n_seq)
        n_batch = x.size(0)
        n_seq = x.size(1)

        # i_seq: (n_batch, n_seq, n_emb)
        # i_seq starts from 1, as paddings will occupy 0
        i_seq = torch.arange(n_seq, device=x.device).expand(n_batch, n_seq) + 1
        i_seq = i_seq.masked_fill(x.eq(0), 0)

        # => (n_batch, n_seq, d_emb)
        return self.embedding(i_seq)