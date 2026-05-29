import torch
import math

from lightning_utilities import StrEnum
from torch import nn

class PositionalEncodingType(StrEnum):
    """Positional encoding types"""
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"

class PositionalEncoding(nn.Module):
    """Positional encoding wrapper"""
    def __init__(self, pe_dim: int, max_len: int, dropout: float,
                 pe_type: PositionalEncodingType):
        super().__init__()
        self.pe = None

        match pe_type:
            case "sinusoidal": self.pe = SinusoidalPositionalEncoding(pe_dim, dropout, max_len)
            case "learned": self.pe = LearnedPositionalEncoding(pe_dim, max_len)
            case "rope": self.pe = RotaryPositionalEncoding(pe_dim, max_len)
            case _:
                raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe(x)

class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding"""
    def __init__(self, pe_dim: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_dim = pe_dim

        div_term = torch.exp(torch.arange(0, pe_dim, 2) * (-math.log(10000.0) / pe_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        pe = torch.zeros(seq_len, self.pe_dim, device=x.device, dtype=x.dtype)
        div_term = self.div_term.to(dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    def __init__(self, pe_dim: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, pe_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        return x + self.embedding(positions)

class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE)"""
    def __init__(self, pe_dim: int, max_len: int):
        super().__init__()

        inv_freq = 1. / (10000 ** (torch.arange(0, pe_dim, 2).float() / pe_dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        position = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        sinusoid_inp = torch.outer(position, self.inv_freq)
        cos = sinusoid_inp.cos().unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
        sin = sinusoid_inp.sin().unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)

        return (x * cos) + (self.rotate_half(x) * sin)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates vectors Q and K in 2D subspace"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
