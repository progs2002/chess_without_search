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
        model_dim: int,
        vocab_size: int,
        seq_len:int = 70
    ):
        super().__init__()

        self.seq_len = seq_len

        self.token_emb_layer = nn.Embedding(vocab_size, model_dim)
        self.pos_emb_layer = nn.Embedding(self.seq_len, model_dim)

        self.positions = nn.Parameter(
            torch.arange(0, self.seq_len),
            requires_grad=False
        )

    def forward(self, x):
        token_emb = self.token_emb_layer(x)
        pos_emb = self.pos_emb_layer(self.positions)

        return token_emb + pos_emb

class SelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        key_dim: int|None = None,
        value_dim: int|None = None, 
        bias: bool = True,
        dropout: float = 0.2
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

        self.dropout_layer = nn.Dropout(dropout)

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


class FFN(nn.Module):
    def __init__(
        self,
        model_dim: int,
        bias: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()

        self.dropout_layer = nn.Dropout(dropout)

        self.fc1 = nn.Linear(model_dim, 4 * model_dim, bias=bias)
        self.fc2 = nn.Linear(4 * model_dim, model_dim, bias=bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return self.dropout_layer(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        key_dim: int|None = None,
        value_dim: int|None = None, 
        bias: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()

        self.l_norm1 = nn.LayerNorm(model_dim)
        self.self_attention_block = SelfAttention(model_dim, n_heads, key_dim, value_dim, bias, dropout)
        self.l_norm2 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, bias, dropout)

    def forward(self, x):
        x = self.l_norm1(self.self_attention_block(x) + x)
        x = self.l_norm2(self.ffn(x) + x)

        return x

class Decoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        n_bins: int = 128,
        seq_len:int = 70,
        key_dim: int|None = None,
        value_dim: int|None = None, 
        bias: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()

        self.emb_layer = EmbeddingLayer(model_dim, vocab_size, seq_len)
        
        self.decoder_blocks = nn.Sequential(*
            [
                DecoderBlock(model_dim, n_heads, key_dim, value_dim, bias, dropout) for _ in range(n_layers)
            ]
        )

        self.classification_head = nn.Linear(model_dim, n_bins, bias)

    def forward(self, x):
        x = self.emb_layer(x)
        x = self.decoder_blocks(x)
        x = self.classification_head(x[:,-1,:])

        return x
        