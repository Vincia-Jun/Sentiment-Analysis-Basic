"""
Author: Jun
Date  : 2024-04-26
"""
import torch
import torch.nn as nn
from Transformer.Embedding import Embedding
from Transformer.PositionalEmbedding import PositionalEmbedding
from Transformer.Encoder import Encoder

class TransformerText(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.position_embedding = PositionalEmbedding(config)
        self.encoder = Encoder(config)
        self.fc = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.position_embedding(out)
        out = self.encoder(out)

        features = out[:, -1, :]
        return self.fc(features)
    


class Config:

    def __init__(self):
        self.input_size = 35093
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 6
        self.dropout = 0.4
        self.max_length = 5000
        self.d_ff = 512
        self.size = self.d_model
        self.num_classes = 2


if __name__ == '__main__':

    config = Config()
    model = TransformerText(config)
    inputs = torch.randint(100,(10,200))
    out = model(inputs)
    print(out.shape)

