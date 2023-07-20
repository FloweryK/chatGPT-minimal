import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # indices
        i_emb = torch.arange(config.d_emb)
        i_seq = torch.arange(config.n_seq).unsqueeze(1)

        # embedding
        self.embedding = i_seq * torch.pow(10000, -2 * i_emb / config.d_emb)
        self.embedding[:, 0::2] = torch.sin(self.embedding[:, 0::2])
        self.embedding[:, 1::2] = torch.cos(self.embedding[:, 1::2])
        self.embedding = nn.Embedding.from_pretrained(self.embedding, freeze=True)
    
    def forward(self, x):
        n_seq = x.size(-1)
        i_seq = torch.arange(n_seq).to(x.device)

        # x: (n_batch, n_seq, d_emb)
        x = self.embedding(i_seq)
        return x


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_embedding = nn.Embedding(config.n_vocab, config.d_emb)
        self.positional_embedding = PositionalEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (n_batch, n_seq)
        
        # x: (n_batch, n_seq, d_emb)
        x = self.input_embedding(x) + self.positional_embedding(x)
        x = self.dropout(x)
        return x
