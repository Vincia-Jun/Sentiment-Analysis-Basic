"""
Author: Jun
Date  : 2024-04-26
"""

import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

        pe = torch.zeros(config.max_length, config.d_model)
        position = torch.arange(0, config.max_length).unsqueeze(1)
        divided_term = torch.exp(torch.arange(0.,config.d_model, 2) * -1 * (math.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * divided_term)
        pe[:, 1::2] = torch.cos(position * divided_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x)


