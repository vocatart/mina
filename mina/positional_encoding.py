import torch
import math
from torch import nn

class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding"""
    def __init__(self, pe_dim, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2) * (-math.log(10000.0) / pe_dim))

        pe = torch.zeros(max_len, 1, pe_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe,)

    def forward(self, x):
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    def __init__(self, pe_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(max_len, pe_dim)

    def forward(self, x):
        return x + self.embedding(torch.arange(x.size(1)).expand(x.size(0), -1))

class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE)"""
    def __init__(self, pe_dim, max_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, pe_dim, 2).float() / pe_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(max_len).float()
        sinusoid_inp = torch.outer(position, inv_freq)

        self.register_buffer('cos', sinusoid_inp.cos())
        self.register_buffer('sin', sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)

        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)

        return (x * cos) + (self.rotate_half(x) * sin)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, pe_dim, max_len):
        super().__init__()
        self.max_len = max_len
        self.relative_attention_bias = nn.Parameter(torch.rand(2 * max_len + 1, pe_dim))

    def forward(self, length):
        context_position = torch.arange(length, dtype=torch.long)[:, None]
        memory_position = torch.arange(length, dtype=torch.long)[None, :]

        relative_position = memory_position - context_position
        relative_position_bucket = relative_position + self.max_len
        return self.relative_attention_bias[relative_position_bucket]