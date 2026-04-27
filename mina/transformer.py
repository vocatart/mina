import torch
import torch.nn as nn
import torch.nn.functional as F

class MinaTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, padding_mask=None):
        b, t, d = x.shape

        attn_bias = None
        if padding_mask is not None:
            attn_bias = torch.zeros(b, 1, 1, t, device=x.device, dtype=x.dtype)
            attn_bias = attn_bias.masked_fill(padding_mask.view(b, 1, 1, t), float('-inf'))

        skip = x
        x = self.norm1(x)
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        dropout_p = self.dropout.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        x = self.out_proj(x.transpose(1, 2).reshape(b, t, d))
        x = skip + self.dropout(x)

        skip = x
        x = self.norm2(x)
        x = skip + self.dropout(self.ffn(x))

        return x


class MinaTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, dropout):
        super().__init__()
        
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.append(MinaTransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout))

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return x
