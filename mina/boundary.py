import torch
from torch import nn

from mina.positional_encoding import PositionalEncodingType
from mina.temporal import TemporalContextEncoder


class BoundaryClassifier(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 temporal_heads: int,
                 temporal_layers: int,
                 temporal_feedforward_dim: int,
                 temporal_dropout: float,
                 max_len: int,
                 pe_type: PositionalEncodingType) -> None:

        super().__init__()
        self.temporal = TemporalContextEncoder(
            hidden_dim,
            temporal_heads,
            temporal_layers,
            temporal_feedforward_dim,
            temporal_dropout,
            max_len,
            pe_type
        )

        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.proj(x)
        return x.squeeze(-1)