import torch.nn as nn

from model.embedding import Embeddings, PositonalEncoding
from model.attention import MultiHeadAttention, SelfAttention
from model.ffnet import FFNet


class EncoderLayer(nn.Module):
    def __init__(self, head=8, model_dim=512, ff_dim=2048, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, head, dropout_rate)
        self.ffnet = FFNet(model_dim, ff_dim, dropout_rate)

    def forward(self, source, source_mask):
        # self attention
        encoded, self_attention = self.self_attention(source, source, source, source_mask)
        # feed forward network
        encoded = self.ffnet(encoded)
        return encoded, self_attention


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        n_position=256,
        n_layers=6,
        head=8,
        model_dim=512,
        ff_dim=2048,
        dropout_rate=0.1,
    ):
        super(Encoder, self).__init__()
        self.embedding = Embeddings(vocab_size, model_dim)
        self.positional_encoding = PositonalEncoding(model_dim, n_position, dropout_rate)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(head, model_dim, ff_dim, dropout_rate)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, source, source_mask):
                                                    # source_mask = source_mask.unsqueeze(-2)
        # apply word embedding(x)
        out = self.embedding(source)
        # apply positional_encoding
        out = self.positional_encoding(out)
        out = self.layer_norm(out)
        for layer in self.layers:
            out, _ = layer(out, source_mask)
        return out
