import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.linear1 = nn.Linear(config.d_emb, config.d_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.d_hidden, config.d_emb)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
