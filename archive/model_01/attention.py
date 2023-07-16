import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.d_hidden = config.d_hidden
        self.scale = config.scale

        self.W_Q = nn.Linear(config.d_emb, config.n_head * config.d_hidden)
        self.W_K = nn.Linear(config.d_emb, config.n_head * config.d_hidden)
        self.W_V = nn.Linear(config.d_emb, config.n_head * config.d_hidden)
        self.W_O = nn.Linear(config.n_head * config.d_hidden, config.d_emb)
    
    def forward(self, q, k, v, mask=None):
        # q, k, v: (n_batch, n_seq, d_emb)
        # mask: (n_batch, n_seq, n_seq)

        # Q, K, V: (n_batch, n_head, n_seq, d_hidden)
        Q = self.W_Q(q).view(q.size(0), -1, self.n_head, self.d_hidden).transpose(1, 2)
        K = self.W_K(k).view(k.size(0), -1, self.n_head, self.d_hidden).transpose(1, 2)
        V = self.W_V(v).view(v.size(0), -1, self.n_head, self.d_hidden).transpose(1, 2)

        # score: (n_batch, n_head, n_seq_q, n_seq_k)
        score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            score = score.masked_fill(mask, -1e9)
        prob = F.softmax(score, dim=-1)

        # context: (n_batch, n_head, n_seq_q, d_hidden)
        context = torch.matmul(prob, V)

        # => (n_batch, n_seq_q, n_head * d_hidden)
        context = context.transpose(1, 2).reshape(q.size(0), -1, self.n_head * self.d_hidden) # TODO: reshape to contiguous

        # => (n_batch, n_seq_q, d_emb)
        context = self.W_O(context)

        return context