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
        skip = x

        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.norm(x_conv)

        x = x_conv + skip
        x = self.activation(x)
        x = self.dropout(x)

        return x

class ConvolutionalAcousticEncoder(nn.Module):
    """Convolutional encoder for extracting mel features"""
    def __init__(self, mel_dim, latent_dim, hidden_dim, num_conv_layers, kernel_size, dropout):
        super().__init__()

        self.mel_dim = mel_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # convert 2d mel to 1d for processing in Conv1D layers
        self.input = nn.Sequential(nn.Linear(mel_dim, hidden_dim), nn.LayerNorm(latent_dim), nn.ReLU())

        self.conv_block = nn.Sequential()
        for i in range(num_conv_layers):
            self.conv_blocks.append(MelConvBlock(latent_dim, kernel_size, dropout))

        self.output = nn.Linear(latent_dim, hidden_dim)

    def forward(self, x):
        x = self.input(x)
        x = self.conv_block(x)
        x = self.output(x)

        return x