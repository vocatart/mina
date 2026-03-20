from torch import nn

class MelConvBlock(nn.Module):
    """Convolutional layer for extracting mel features"""
    def __init__(self, latent_dim, kernel_size, dropout):
        super().__init__()

        self.conv = nn.Conv1d(latent_dim, latent_dim, kernel_size, padding="same")
        self.norm = nn.LayerNorm(latent_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.norm(x_conv)

        x = x_conv + residual
        x = self.activation(x)
        x = self.dropout(x)

        return x