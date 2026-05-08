import torch
from torch import nn

class MelConvBlock(nn.Module):
    """Convolutional layer for extracting mel features"""
    def __init__(self, latent_dim: int, kernel_size: int, dropout: float,
                 dilation: int, reduction: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(latent_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size, padding="same", dilation=dilation, groups=latent_dim),
            nn.Conv1d(latent_dim, latent_dim, 1)
        )

        self.sne = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // reduction),
            nn.ReLU(),
            nn.Linear(latent_dim // reduction, reduction),
            nn.Sigmoid()
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x

        x_conv = self.norm(x)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)

        sne = self.sne(x.mean(dim=1, keepdim=True))
        x_conv = sne * x_conv

        x = x_conv + skip
        x = self.activation(x)
        x = self.dropout(x)

        return x

class ConvAcousticEncoder(nn.Module):
    """Convolutional encoder for extracting mel features"""
    def __init__(self, mel_dim: int, latent_dim: int, hidden_dim: int,
                 num_conv_layers: int, kernel_size: int, dropout: float,
                 num_heads: int) -> None:
        super().__init__()

        # project mel features to the convolution width
        self.input = nn.Sequential(
            nn.Linear(mel_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

        self.conv_block = nn.Sequential(*[
            MelConvBlock(latent_dim, kernel_size, dropout, dilation=2**(i%4), reduction=4) for i in range(num_conv_layers)
        ])

        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(latent_dim)

        self.output = nn.Linear(latent_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.conv_block(x)

        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + attn_out)

        x = self.output(x)
        return x
