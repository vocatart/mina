import torch
from torch import nn
from mina.positional_encoding import PositionalEncoding, PositionalEncodingType
from mina.transformer import MinaTransformerEncoder


class BoundaryDetector(nn.Module):
    """Encoder-only Transformer Boundary Detector"""
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int,
                 feedforward_dim: int, dropout: float, max_len: int,
                 pe_type: PositionalEncodingType) -> None:
        super().__init__()

        self.positional_encoding = PositionalEncoding(hidden_dim, max_len, dropout, pe_type)

        self.transformer = MinaTransformerEncoder(
            hidden_dim,
            num_heads,
            num_layers,
            feedforward_dim,
            dropout,
        )

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        x = self.positional_encoding(x)
        x = self.transformer(x, padding_mask=padding_mask)

        return self.output(x).squeeze(-1)
