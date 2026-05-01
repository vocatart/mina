from torch import nn


class PhonemeDetector(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, vocab_size: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(latent_dim, vocab_size)

    def forward(self, x):
        x = self.attention(x)
        x = self.out(x)
        return x