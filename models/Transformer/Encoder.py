"""
Author: Jun
Date  : 2024-04-26
"""

import torch.nn as nn
from .EncoderLayer import EncoderLayer
import copy

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder_layer = EncoderLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.size)

    def forward(self, x, mask = None):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)



