import math
import torch
import torch.nn as nn
from einops import rearrange
from .lora import LoRALinear


def _linear(in_f: int, out_f: int, use_lora: bool, rank: int) -> nn.Module:
    return LoRALinear(in_f, out_f, rank=rank) if use_lora else nn.Linear(in_f, out_f)


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention.
    Supports self-attention (context=None) and cross-attention (context given).
    Optionally wraps projections with LoRA for Stage-2 fine-tuning.
    """

    def __init__(self, dim: int, num_heads: int,
                 use_lora: bool = False, lora_rank: int = 64):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.q   = _linear(dim, dim, use_lora, lora_rank)
        self.k   = _linear(dim, dim, use_lora, lora_rank)
        self.v   = _linear(dim, dim, use_lora, lora_rank)
        self.out = _linear(dim, dim, use_lora, lora_rank)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        kv_src = context if context is not None else x
        q = rearrange(self.q(x),      'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k(kv_src), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v(kv_src), 'b n (h d) -> b h n d', h=self.num_heads)

        scale = math.sqrt(self.head_dim)
        attn  = torch.softmax(q @ k.transpose(-2, -1) / scale, dim=-1)
        out   = rearrange(attn @ v, 'b h n d -> b n (h d)')
        return self.out(out)
