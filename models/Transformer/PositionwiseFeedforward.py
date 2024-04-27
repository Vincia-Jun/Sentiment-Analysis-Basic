"""
Author: Jun
Date  : 2024-04-26
"""

import torch
import torch.nn as nn

class PositionwiseFeedforward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.d_model, config.d_ff)
        self.fc_2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = self.dropout(torch.relu(self.fc_1(x)))

        return self.fc_2(out)

