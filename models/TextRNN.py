"""
Author: Jun
Date  : 2024-04-26
"""

import torch
import torch.nn as nn
from sklearn import metrics

class TextRNN(nn.Module):

    def __init__(self, vocab_size, emb_dim, hidden_size, num_classes, num_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, bidirectional=bidirectional, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*(2 if bidirectional else 1), num_classes)

    def forward(self, x):
        
        x = x.permute(1, 0)                    # x = [seq_len, batch_size]
        out = self.dropout(self.embedding(x))  # out = [seq_len, batch_size, emb_dim]
        out,_ = self.rnn(out)                  # out = [seq_len, batch_size, num_directions * hidden_size]
        out = self.fc(out[-1,:,:])             # out是整句话每个单词输出的hidden state，取最后一个单词的时候所得到的输出
        return out

if __name__ == '__main__':

    vocab_len = 35093
    embedding_dim = 300
    model = TextRNN(vocab_size = vocab_len,
                    emb_dim = embedding_dim,
                    hidden_size = 128,
                    num_classes = 2,
                    num_layers = 2,
                    bidirectional = True,
                    dropout = 0.4)

    inputs = torch.randint(100,(10, 200)) # [B, seq_len]
    output = model(inputs)
    print(output.shape)                   # [B, num_classes]