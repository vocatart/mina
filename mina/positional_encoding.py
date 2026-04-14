import torch
import math

from lightning_utilities import StrEnum
from torch import nn

class PositionalEncodingType(StrEnum):
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"

class PositionalEncoding(nn.Module):
    def __init__(self, pe_dim, max_len, dropout, pe_type: PositionalEncodingType):
        super().__init__()
        self.pe = None

        match pe_type:
            case "sinusoidal": self.pe = SinusoidalPositionalEncoding(pe_dim, dropout, max_len)
            case "learned": self.pe = LearnedPositionalEncoding(pe_dim, max_len)
            case "rope": self.pe = RotaryPositionalEncoding(pe_dim, max_len)
            case _:
                raise NotImplementedError

    def forward(self, x):
        return self.pe(x)



class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding"""
    def __init__(self, pe_dim, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2) * (-math.log(10000.0) / pe_dim))

        pe = torch.zeros(1, max_len, pe_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    def __init__(self, pe_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(max_len, pe_dim)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        return x + self.embedding(positions)

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

        if seq_len > self.cos.size(0):
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds positional encoding max_len ({self.cos.size(0)}). "
                "Regenerate binarized data with a larger max_len or reduce sequence length."
            )

        cos = self.cos[:seq_len].unsqueeze(0).to(dtype=x.dtype)
        sin = self.sin[:seq_len].unsqueeze(0).to(dtype=x.dtype)

        return (x * cos) + (self.rotate_half(x) * sin)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
