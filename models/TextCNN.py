"""
Author: Jun
Date  : 2024-04-26
"""
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, filter_sizes, num_classes, vocab_size, num_filters, emb_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, x) for x in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.relu = nn.ReLU()

    def pool(self, out, conv):
        out = self.relu(conv(out))
        max_pool = nn.MaxPool1d(out.shape[-1])
        out = max_pool(out)
        out = out.squeeze(2)
        return out

    def forward(self, x):

        embedded = self.dropout(self.embedding(x))      # x = [batch_size, seq_len]
        embedded = embedded.permute(0,2,1)              # embedded = [batch_size, seq_len, emb_dim]
        output = [self.pool(embedded, conv) for conv in self.convs]
        out = torch.cat(output, dim=1)
        out = self.fc(out)
        return out

if __name__ == '__main__':

    vocab_len = 35093
    embedding_dim = 300
    model = TextCNN(filter_sizes = [3,4,5],
                    num_classes = 2,
                    vocab_size = vocab_len,
                    num_filters = 128,
                    emb_dim = embedding_dim,
                    dropout = 0.4)

    inputs = torch.randint(100,(10, 200))
    output = model(inputs)
    print(output.shape)