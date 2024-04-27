"""
Author: Jun
Date  : 2024-04-26
"""
import torch.nn as nn
from .ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_heads = self.d_model // self.n_heads
        self.attn = ScaledDotProductAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_Q = nn.Linear(config.d_model, config.d_model)
        self.fc_K = nn.Linear(config.d_model, config.d_model)
        self.fc_V = nn.Linear(config.d_model, config.d_model)
        self.fc_O = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, mask = None):
        
        Q = self.fc_Q(x)            # Q: [batch_size, Q_len, d_model]
        K = self.fc_K(x)            # K: [batch_size, K_len, d_model]
        V = self.fc_V(x)            # V: [batch - size, V_len, d_model]

        batch_size = Q.shape[0]

        Q = Q.view(batch_size, -1, self.n_heads, self.d_heads).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.d_heads).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.d_heads).permute(0,2,1,3)

        out = self.attn(Q, K, V, mask)                                              # out = [batch_size, n_heads, query_len, d_heads]
        out = out.permute(0,2,1,3).contiguous().view(batch_size, -1, self.d_model)  # out = [batch_size, query_len, d_model]
        out = self.fc_O(out)
        return out



