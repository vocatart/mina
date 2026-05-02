import torch
from torch import nn


class PhonemeClassifier(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # frame & seg
        self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.frame_out = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

        # seg
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seg_out = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, gt_boundaries: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None):

        # (B, T, d_h)
        x = self.hidden_projection(x)
        x = self.hidden_norm(x)
        x = self.relu(x)

        # (B, T, v)
        frame_logits = self.frame_out(x)

        if gt_boundaries is None:
            return frame_logits, None

        seg_cumulated = gt_boundaries.cumsum(dim=-1) # (B, T)
        num_segs = int(seg_cumulated.max().item()) + 1
        seg_onehot = torch.nn.functional.one_hot(seg_cumulated, num_segs) # (B, T, num_segs)

        if padding_mask is not None:
            seg_onehot = seg_onehot * (~padding_mask).unsqueeze(-1).float()

        counts = seg_onehot.sum(dim=1, keepdim=True).clamp(min=1)
        seg_weights = seg_onehot / counts # (B, T, num_segs)

        # for each segment s take a weighted average of all frame features
        seg_repr = torch.einsum('bts,btd->bsd', seg_weights, x) # (B, num_segs, d_h)

        q = self.q_proj(seg_repr)
        k = self.k_proj(x)
        v = self.v_proj(x)
        dropout_p = self.dropout.p if self.training else 0.0
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        seg_logits = self.seg_out(attn)

        return frame_logits, seg_logits
