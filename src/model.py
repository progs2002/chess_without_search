import dataclasses

import math

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange

@dataclasses.dataclass()
class ModelConfig:
    model_dim: int = 64
    n_layers: int = 8
    n_heads: int = 4
    vocab_size: int = 42
    seq_len: int = 70
    key_dim: int|None = None
    value_dim: int|None = None
    dropout: float = 0
    bias: bool = True
    n_bins: int = 32

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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

        queries = rearrange(queries, 'b c (h k) -> b h c k', h=self.n_heads)
        keys = rearrange(keys, 'b c (h k) -> b h c k', h=self.n_heads)
        values = rearrange(values, 'b c (h v) -> b h c v', h=self.n_heads)

        attention = F.softmax(
            (queries @ torch.transpose(keys, -1,-2)) / math.sqrt(self.key_dim),
            dim = -1
        )

        out = attention @ values

        out_concat = rearrange(out, 'b h c v -> b c (h v)')

        out_fc = self.fc(out_concat)

        return self.dropout_layer(out_fc)


class FFN(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.dropout_layer = nn.Dropout(config.dropout)

        self.fc_expand = nn.Linear(config.model_dim, 4 * config.model_dim, bias=config.bias)
        self.fc = nn.Linear(4 * config.model_dim, config.model_dim, bias=config.bias)

        self.silu = nn.SiLU()
    
    def forward(self, x):
        x = self.fc_expand(x)
        x = self.silu(x)
        x = self.fc(x)

        return self.dropout_layer(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.l_norm1 = LayerNorm(config.model_dim, bias=config.bias)
        self.self_attention_block = SelfAttention(config)

        self.l_norm2 = LayerNorm(config.model_dim, bias=config.bias)
        self.ffn = FFN(config)

    def forward(self, x):
        x = self.self_attention_block(self.l_norm1(x)) + x
        x = self.ffn(self.l_norm2(x)) + x

        return x

class Decoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.emb_layer = EmbeddingLayer(config)
        
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(config) for _ in range(config.n_layers)
            ]
        )

        self.post_l_norm = LayerNorm(config.model_dim, config.bias)

        self.classification_head = nn.Linear(
            config.model_dim, 
            config.n_bins, 
            config.bias
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('fc.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.emb_layer(x)

        for block in self.decoder_blocks:
            x = block(x)
            
        x = self.post_l_norm(x)
        x = self.classification_head(x)

        return x
        