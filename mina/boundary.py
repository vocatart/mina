import torch
from torch import nn
from mina.positional_encoding import SinusoidalPositionalEncoding

class BoundaryDetector(nn.Module):
    """Encoder-only Transformer Boundary Detector"""
    def __init__(self, hidden_dim, num_heads, num_layers, feedforward_dim, dropout, max_len):
        super().__init__()

        # TODO: test the other ones? Rotary will probably work best here
        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim, dropout, max_len)

        # right now i just need this model to WORK so im using the unoptimized pytorch stuff
        # TODO: swap this part out for something that is more memory efficient (xformers blocks?)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )

        # TODO: for some reason pytorch makes it so you have to MANUALLY INITIALIZE ALL LAYERS???
        # I've done this before in other work, I just need to port it over

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, padding_mask=None):
        x = self.positional_encoding(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        return self.output(x).squeeze(-1)
