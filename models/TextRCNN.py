"""
Author: Jun
Date  : 2024-04-26
"""
import torch
import torch.nn as nn

class TextRCNN(nn.Module):

    def __init__(self, vocab_size, emb_dim, hidden_size, num_classes, dropout, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, bidirectional=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.w = nn.Parameter(torch.randn(2*hidden_size + emb_dim, 2 * hidden_size))

    def forward(self, X):
        
        X = X.permute(1, 0)                 # X = [seq_len, batch_size]
        embedded = self.embedding(X)        # embedded = [seq_len, batch_size, hidden_size]
        
        out,_ = self.rnn(embedded)
        out = out.permute(1,0,2)                     # out = [seq_len, batch_size, hidden_size * 2]
        embedded = embedded.permute(1,0,2)
        out = torch.cat((out, embedded), dim=2)      # out = [batch_size, seq_len, hidden_size * 2 + emb_dim
        out = torch.tanh(torch.matmul(out, self.w))  # out = [batch_size, seq_len, hidden_size * 2
        out = out.permute(0,2,1)                     # out = [batch_size, hidden_size * 2, seq_len]
        
        out = nn.functional.max_pool1d(out, out.shape[-1]).squeeze(2)
        out = self.fc(out)
        return out

if __name__ == '__main__':

    vocab_len = 35093
    embedding_dim = 300
    inputs = torch.randint(100,(10,128))
    model = TextRCNN(vocab_size = vocab_len,
                     emb_dim = embedding_dim,
                     hidden_size = 256,
                     num_classes = 2,
                     dropout = 0.4,
                     num_layers = 1)
    
    inputs = torch.randint(100,(10, 200))
    output = model(inputs)
    print(output.shape)