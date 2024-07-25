import dataclasses

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange

@dataclasses.dataclass()
class ModelConfig:
    model_dim: int = 64
    n_layers: int = 32
    n_heads: int = 4
    vocab_size: int = 42
    seq_len: int = 70
    key_dim: int|None = None
    value_dim: int|None = None
    dropout: float = 0
    bias: bool = True
    n_bins: int = 32

class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.token_emb_layer = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_emb_layer = nn.Embedding(config.seq_len, config.model_dim)

        self.positions = nn.Parameter(
            torch.arange(0, config.seq_len),
            requires_grad=False
        )

    def forward(self, x):
        token_emb = self.token_emb_layer(x)
        pos_emb = self.pos_emb_layer(self.positions)

        return token_emb + pos_emb

class SelfAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.n_heads = config.n_heads

        assert config.model_dim % config.n_heads == 0

        if config.value_dim is None:
            self.value_dim = config.model_dim // config.n_heads
        else:
            self.value_dim = config.value_dim

        if config.key_dim is None:
            self.key_dim = self.value_dim
        else:
            self.key_dim = config.key_dim

        self.dropout_layer = nn.Dropout(config.dropout)

        self.query_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=self.n_heads * self.key_dim,
            bias=config.bias
        )

        self.key_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=self.n_heads * self.key_dim,
            bias=config.bias
        )

        self.value_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=self.n_heads * self.value_dim,
            bias=config.bias
        )

        self.fc = nn.Linear(
            in_features=self.n_heads * self.value_dim,
            out_features=config.model_dim,
            bias=config.bias
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
        config: ModelConfig
    ):
        super().__init__()

        self.dropout_layer = nn.Dropout(config.dropout)

        self.fc1 = nn.Linear(config.model_dim, 4 * config.model_dim, bias=config.bias)
        self.fc2 = nn.Linear(4 * config.model_dim, config.model_dim, bias=config.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return self.dropout_layer(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.l_norm1 = nn.LayerNorm(config.model_dim)
        self.self_attention_block = SelfAttention(config)

        self.l_norm2 = nn.LayerNorm(config.model_dim)
        self.ffn = FFN(config)

    def forward(self, x):
        x = self.l_norm1(self.self_attention_block(x) + x)
        x = self.l_norm2(self.ffn(x) + x)

        return x

class Decoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.emb_layer = EmbeddingLayer(config)
        
        self.decoder_blocks = nn.Sequential(*
            [
                DecoderBlock(config) for _ in range(config.n_layers)
            ]
        )

        self.classification_head = nn.Linear(
            config.model_dim, 
            config.n_bins, 
            config.bias
        )

    def forward(self, x):
        x = self.emb_layer(x)
        x = self.decoder_blocks(x)
        x = self.classification_head(x[:,-1,:])

        return x
        