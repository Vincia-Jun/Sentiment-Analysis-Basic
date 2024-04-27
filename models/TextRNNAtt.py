"""
Author: Jun
Date  : 2024-04-26
"""
import torch
import torch.nn as nn

class TextRNNAtt(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size1, hidden_size2, dropout, bidirectional, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size1, bidirectional=bidirectional, num_layers = num_layers
                           ,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.w = nn.Parameter(torch.randn(2 * hidden_size1))
        self.fc1 = nn.Linear(hidden_size1*2, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, num_classes)
        
    
    def forward(self, X):
                                                            # X = [batch_size, seq_len]
        embedded = self.dropout(self.embedding(X))          # embedded = [batch_size, seq_len, emb_dim]
        
        out,(H,C) = self.rnn(embedded)
        # out = [batch_size, seq_len, hidden_size * num_directions]
        # H = [num_layers * num_direction, batch_size, hidden_size]
        # C = [num_layers * num_direction, batch_size, hidden_size]

        # Attention Stage
        M = self.tanh1(out)
        score = torch.matmul(out, self.w)
        att = torch.softmax(score, dim=1).unsqueeze(2)      # att = [batch_size, seq_len]
        out = out * att                                     # out = [batch_size, seq_len, hidden_size * 2]
        out = torch.sum(out, 1)                             # out = [batch_size, hidden_size * 2]
        out = torch.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


if __name__ == '__main__':

    vocab_len = 35093
    embedding_dim = 300
    model = TextRNNAtt(vocab_size = vocab_len,
                        emb_dim = embedding_dim,
                        hidden_size1 = 128,
                        hidden_size2 = 64,
                        num_classes = 2,
                        num_layers = 2,
                        bidirectional = True,
                        dropout = 0.4)

    inputs = torch.randint(100,(10,200))
    out = model(inputs)
    print(out.shape)

    