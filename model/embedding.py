import math
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)            # Add pad_idx
        self.scale_factor = math.sqrt(model_dim)

    def forward(self, x:torch.Tensor) -> torch.FloatTensor:
        x = self.embedding(x) * self.scale_factor
        return x
        

class PositonalEncoding(nn.Module):
    def __init__(self, model_dim, n_position=256, dropout_rate=0.1):
        super(PositonalEncoding, self).__init__()
        self.register_buffer('position_table', \
            self.get_sinusoid_encoding_table(model_dim, n_position))
        self.dropout = nn.Dropout(dropout_rate)

    def get_sinusoid_encoding_table(self, model_dim, n_position):
        position_table = torch.zeros(n_position, model_dim)
        position = torch.arange(0, n_position).float().unsqueeze(1)
        div_term = 10000 ** (torch.arange(0.0, model_dim, 2) / model_dim)
        position_table[:, 0::2] = torch.sin(position / div_term)
        position_table[:, 1::2] = torch.cos(position / div_term)
        position_table = position_table.unsqueeze(0)

        return position_table

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        x = x + self.position_table[:, x.size(1), :]
        x = self.dropout(x)
        return x
