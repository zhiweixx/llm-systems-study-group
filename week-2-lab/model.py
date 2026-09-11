"""Small causal decoder for systems experiments, with random weights and a KV cache.

The cache is a preallocated tensor per layer, not repeated torch.cat allocations.
No tokenizer, downloads, dropout, sampling distribution, or serving framework.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 8
    width: int = 1024
    heads: int = 16
    mlp_width: int = 4096
    vocab_size: int = 4096
    max_sequence: int = 4096

    def __post_init__(self):
        if min(vars(self).values()) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, cache=None, start_pos=0):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, self.head_width)
                   .transpose(1, 2) for t in (q, k, v)]
        if cache is not None:
            end = start_pos + length
            if end > cache[0].shape[2]:
                raise ValueError("Token positions exceed the allocated KV-cache capacity")
            cache[0][:, :, start_pos:end, :].copy_(k)
            cache[1][:, :, start_pos:end, :].copy_(v)
            if start_pos:
                if length != 1:
                    raise ValueError("Cached decode accepts exactly one new token")
                k, v = cache[0][:, :, :end, :], cache[1][:, :, :end, :]
        # Prefill is square causal attention. During one-token decode, the cache
        # contains only past/current tokens, so ALL exposed keys are legal.
        # is_causal=True with Lq=1, Lkv>1 is upper-left aligned in PyTorch;
        # it would incorrectly let the new query see only the first key.
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                          is_causal=(start_pos == 0))
        return self.proj(y.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(nn.Linear(config.width, config.mlp_width, bias=False),
                                 nn.GELU(),
                                 nn.Linear(config.mlp_width, config.width, bias=False))

    def forward(self, x, cache=None, start_pos=0):
        x = x + self.attention(self.norm1(x), cache, start_pos)
        return x + self.mlp(self.norm2(x))


class TinyDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.max_sequence, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)

    def allocate_cache(self, batch, capacity=None):
        capacity = self.config.max_sequence if capacity is None else capacity
        if not 1 <= capacity <= self.config.max_sequence:
            raise ValueError("Cache capacity must fit the model's position range")
        shape = (batch, self.config.heads, capacity,
                 self.config.width // self.config.heads)
        weight = self.embedding.weight
        return [(torch.empty(shape, device=weight.device, dtype=weight.dtype),
                 torch.empty(shape, device=weight.device, dtype=weight.dtype))
                for _ in self.blocks]

    def forward(self, tokens, caches=None, start_pos=0, return_all=False):
        length = tokens.shape[1]
        if start_pos < 0 or start_pos + length > self.config.max_sequence:
            raise ValueError("Token positions exceed model/cache capacity")
        if start_pos and caches is None:
            raise ValueError("A nonzero start_pos requires a populated KV cache")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("One KV cache pair is required per layer")
        positions = torch.arange(start_pos, start_pos + length, device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)
        for index, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[index], start_pos)
        # A generation step needs logits only for the last input position.
        # Every prompt position still participates in attention and fills KV.
        x = x if return_all else x[:, -1:, :]
        return self.lm_head(self.norm(x))


@torch.inference_mode()
def check_correctness():
    """CPU FP32: compare cached logits AND greedy generation to full recomputation."""
    torch.manual_seed(2026)
    config = ModelConfig(layers=2, width=32, heads=4, mlp_width=64,
                         vocab_size=41, max_sequence=24)
    model = TinyDecoder(config).float().eval()
    tokens = torch.randint(config.vocab_size, (2, 13))
    full = model(tokens, return_all=True)
    worst = 0.0
    # Multiple prompt lengths catch incorrect position indexing and decode masks.
    for prompt_length in (1, 4, 9):
        cache = model.allocate_cache(tokens.shape[0], capacity=tokens.shape[1])
        prefill = model(tokens[:, :prompt_length], cache, return_all=True)
        torch.testing.assert_close(prefill, full[:, :prompt_length], rtol=1e-5, atol=1e-5)
        for pos in range(prompt_length, tokens.shape[1]):
            cached = model(tokens[:, pos:pos + 1], cache, start_pos=pos)
            expected = full[:, pos:pos + 1]
            worst = max(worst, (cached - expected).abs().max().item())
            torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-5)
    prompt = tokens[:, :4]
    cache = model.allocate_cache(prompt.shape[0], capacity=prompt.shape[1] + 5)
    next_token = model(prompt, cache).argmax(dim=-1)
    full_sequence = prompt.clone()
    for step in range(6):
        expected = model(full_sequence).argmax(dim=-1)
        torch.testing.assert_close(next_token, expected, rtol=0, atol=0)
        full_sequence = torch.cat((full_sequence, expected), dim=1)
        if step < 5:
            next_token = model(next_token, cache, start_pos=prompt.shape[1] + step).argmax(dim=-1)
    return {"device": "cpu", "dtype": "float32", "max_abs_logit_error": worst,
            "cached_logits": "pass", "greedy_generation": "pass"}
