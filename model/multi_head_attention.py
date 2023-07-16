import torch.nn as nn
import config
from model.scaled_dot_produt_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.w_q = nn.Linear(config.d_emb, config.d_emb)
        self.w_k = nn.Linear(config.d_emb, config.d_emb)
        self.w_v = nn.Linear(config.d_emb, config.d_emb)
        self.w_o = nn.Linear(config.d_emb, config.d_emb)
        self.attention = ScaledDotProductAttention()
    
    def forward(self, q, k, v, mask):
        # q: (n_batch, n_seq_q, n_emb)
        # k: (n_batch, n_seq_k, n_emb)
        # v: (n_batch, n_seq_v, n_emb)

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # q: (n_batch, n_head, n_seq_q, d_emb // n_head)
        # k: (n_batch, n_head, n_seq_k, d_emb // n_head)
        # v: (n_batch, n_head, n_seq_v, d_emb // n_head)
        q = q.view(q.size(0), q.size(1), config.n_head, q.size(2) // config.n_head).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), config.n_head, k.size(2) // config.n_head).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), config.n_head, v.size(2) // config.n_head).transpose(1, 2)

        # context: (n_batch, n_head, n_seq_q, d_emb // n_head)
        context = self.attention(q, k, v, mask)

        # SUSPICOUS
        # context: (n_batch, n_seq_q, d_emb)
        context = context.transpose(1, 2).contiguous().view(context.size(0), context.size(2), config.d_emb)
        context = self.w_o(context)

        return context