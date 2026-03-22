import torch
import math
from torch import nn

class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""
    def __init__(self, pe_dim, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2) * (-math.log(10000.0) / pe_dim))

        pe = torch.zeros(max_len, 1, pe_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, pe_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(max_len, pe_dim)

    def forward(self, x):
        return x + self.embedding(torch.arange(x.size(1)).expand(x.size(0), -1))