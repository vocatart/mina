from torch import nn
from mina.positional_encoding import PositionalEncoding
from mina.transformer import MinaTransformerEncoder, MinaTransformerEncoderLayer


class BoundaryDetector(nn.Module):
    """Encoder-only Transformer Boundary Detector"""
    def __init__(self, hidden_dim, num_heads, num_layers, feedforward_dim, dropout, max_len, pe_type):
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

    def forward(self, x, padding_mask=None):
        x = self.positional_encoding(x)
        x = self.transformer(x, padding_mask=padding_mask)

        return self.output(x).squeeze(-1)
