"""Cross-modal attention used by the compact MPDD MulT network."""

import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query, source, source_valid_mask=None):
        key_padding_mask = None if source_valid_mask is None else ~source_valid_mask
        output, _ = self.attention(
            query, source, source, key_padding_mask=key_padding_mask, need_weights=False
        )
        return output
