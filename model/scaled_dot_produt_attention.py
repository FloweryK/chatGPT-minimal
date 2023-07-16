import torch
import torch.nn as nn
import config


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask):
        # q: (n_batch, n_head, n_seq_q, n_emb // n_head)
        # k: (n_batch, n_head, n_seq_k, n_emb // n_head)
        # v: (n_batch, n_head, n_seq_v, n_emb // n_head)

        # score: (n_batch, n_head, n_seq_q, n_seq_k)
        score = torch.matmul(q, k.transpose(-1, -2)) / config.scale
        if mask is not None:
            score = score.masked_fill(mask, -1e9)
        
        score = self.softmax(score)
        context = torch.matmul(score, v)

        return context
