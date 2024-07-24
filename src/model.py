import dataclasses

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange

@dataclasses.dataclass()
class ModelConfig:
    n_layers: int = 32
    n_bins: int = 128

class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        seq_len:int = 70
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.seq_len = seq_len

        self.token_emb_layer = nn.Embedding(self.vocab_size, self.model_dim)
        self.pos_emb_layer = nn.Embedding(self.seq_len, self.model_dim)

    def forward(self, x):
        token_emb = self.token_emb_layer(x)
        pos_emb = self.pos_emb_layer(
            torch.arange(0, self.seq_len)
        )

        return token_emb + pos_emb

class SelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        key_dim: int|None = None,
        value_dim: int|None = None, 
        bias: bool = True,
        dropout = 0.2
    ):
        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads

        assert model_dim % n_heads == 0

        if value_dim is None:
            self.value_dim = model_dim // n_heads
        else:
            self.value_dim = value_dim

        if key_dim is None:
            self.key_dim = self.value_dim
        else:
            self.key_dim = key_dim

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        self.query_proj = nn.Linear(
            in_features=self.model_dim,
            out_features=self.n_heads * self.key_dim,
            bias=bias
        )

        self.key_proj = nn.Linear(
            in_features=self.model_dim,
            out_features=self.n_heads * self.key_dim,
            bias=bias
        )

        self.value_proj = nn.Linear(
            in_features=model_dim,
            out_features=self.n_heads * self.value_dim,
            bias=bias
        )

        self.fc = nn.Linear(
            in_features=self.n_heads * self.value_dim,
            out_features=self.model_dim,
            bias=bias
        )

    def forward(self, x):
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        queries = rearrange(queries, 'b c (h k) -> b c h k', h=self.n_heads)
        keys = rearrange(keys, 'b c (h k) -> b c h k', h=self.n_heads)
        values = rearrange(values, 'b c (h v) -> b c h v', h=self.n_heads)

        attention = F.softmax(
            (queries @ torch.transpose(keys, -1,-2)) / self.key_dim,
            dim = -1
        )

        out = attention @ values

        out_concat = rearrange(out, 'b c h v -> b c (h v)')

        out_fc = self.fc(out_concat)

        return self.dropout_layer(out_fc)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        T: int
    ):
        pass

        

if __name__ == "__main__":
    model = EmbeddingLayer(42, 512)

    from src.utils import CustomDataLoader
    loader = CustomDataLoader('../data/test.csv')

    x_test = next(iter(loader))
    output = model(x_test)

    print(output.shape)