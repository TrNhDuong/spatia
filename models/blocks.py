import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .lora import LoRALinear


def _linear(in_f: int, out_f: int, use_lora: bool, rank: int) -> nn.Module:
    return LoRALinear(in_f, out_f, rank=rank) if use_lora else nn.Linear(in_f, out_f)


# ─────────────────────────────────────────────────────────
# FFN
# ─────────────────────────────────────────────────────────
class FFN(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 use_lora: bool = False, lora_rank: int = 64):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = _linear(dim, hidden, use_lora, lora_rank)
        self.fc2 = _linear(hidden, dim, use_lora, lora_rank)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ─────────────────────────────────────────────────────────
# Main Block  (Wan2.2-style DiT block)
# ─────────────────────────────────────────────────────────
class MainBlock(nn.Module):
    """
    One DiT-style transformer block:
        x → LayerNorm → Self-Attention   → residual
          → LayerNorm → Cross-Attention  → residual   (text as key/value)
          → LayerNorm → FFN              → residual
          + scene_cond  (additive, from paired ControlNet block)
    """

    def __init__(self, dim: int, num_heads: int, text_dim: int,
                 mlp_ratio: float = 4.0,
                 use_lora: bool = False, lora_rank: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn  = MultiHeadAttention(dim, num_heads,
                                             use_lora=use_lora, lora_rank=lora_rank)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads,
                                             use_lora=use_lora, lora_rank=lora_rank)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio, use_lora=use_lora, lora_rank=lora_rank)
        self.text_proj = (nn.Linear(text_dim, dim)
                          if text_dim != dim else nn.Identity())

    def forward(self, x: torch.Tensor, text_tokens: torch.Tensor,
                scene_cond: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x),
                                 context=self.text_proj(text_tokens))
        x = x + self.ffn(self.norm3(x))
        if scene_cond is not None:
            x = x + scene_cond
        return x


# ─────────────────────────────────────────────────────────
# ControlNet Block
# ─────────────────────────────────────────────────────────
class ControlNetBlock(nn.Module):
    """
    Mirror of MainBlock that processes scene point-cloud tokens.
    Appends an MLP projector after FFN whose output is injected
    additively into the paired main block.

    Paper: "each ControlNet block adopts the same architecture
            but appends a projector (simple MLP layer) after the FFN."
    """

    def __init__(self, dim: int, num_heads: int, text_dim: int,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn  = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio)
        self.text_proj = (nn.Linear(text_dim, dim)
                          if text_dim != dim else nn.Identity())
        # MLP projector → produces additive conditioning signal
        self.projector = nn.Linear(dim, dim)

    def forward(self, scene_tokens: torch.Tensor,
                text_tokens: torch.Tensor):
        """
        Returns:
            scene_out  : updated scene token sequence (passed to next ControlNet block)
            projected  : conditioning signal injected into paired main block
        """
        x = scene_tokens
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x),
                                 context=self.text_proj(text_tokens))
        x = x + self.ffn(self.norm3(x))
        return x, self.projector(x)


# ─────────────────────────────────────────────────────────
# Spatia Network Block  (1 ControlNet + N_sub Main blocks)
# ─────────────────────────────────────────────────────────
class SpatiaNetworkBlock(nn.Module):
    """
    One 'network block' as described in the paper:
        - 1 ControlNet block  (processes scene tokens in parallel)
        - num_sub_blocks main blocks  (process video tokens)

    The ControlNet output is injected additively into the first
    main block's x_P and x_t token positions.
    """

    def __init__(self, dim: int, num_heads: int, text_dim: int,
                 mlp_ratio: float = 4.0, num_sub_blocks: int = 4,
                 use_lora: bool = False, lora_rank: int = 64):
        super().__init__()
        self.controlnet = ControlNetBlock(dim, num_heads, text_dim, mlp_ratio)
        self.main_blocks = nn.ModuleList([
            MainBlock(dim, num_heads, text_dim, mlp_ratio,
                      use_lora=use_lora, lora_rank=lora_rank)
            for _ in range(num_sub_blocks)
        ])
        self.controlnet.load_state_dict(self.main_blocks[0].state_dict(), strict=False)

    def forward(self, x_tokens: torch.Tensor,
                scene_tokens: torch.Tensor,
                text_tokens: torch.Tensor,
                n_R: int, n_P: int, n_T: int):
        """
        Args:
            x_tokens    : [B, N_R+N_P+N_T, D]  concat(X_R, X_P, x_t)
            scene_tokens: [B, N_S_P+N_S_T, D]  concat(X_S_P, X_S_T)
                          where N_S_P == n_P and N_S_T == n_T
            text_tokens : [B, N_txt, text_dim]
            n_R, n_P, n_T : token counts for reference / preceding / target parts
        """
        scene_out, projected = self.controlnet(scene_tokens, text_tokens)

        # Split projected into preceding (S_P) and target (S_T) parts.
        # N_S_P == n_P (preceding token count) and N_S_T == n_T (target token count).
        # We must NOT use scene_tokens.shape[1] // 2 because N_P != N_T in general
        # (e.g. preceding_frames=9 → N_P=2400, target_frames=81 → N_T=24000).
        n_s_P = n_P   # scene_P token count equals x_P token count
        n_s_T = n_T   # scene_T token count equals x_T token count
        cond_P = projected[:, :n_s_P, :]        # X'_{S_P}  [B, n_P, D]
        cond_T = projected[:, n_s_P:n_s_P + n_s_T, :]  # X'_{S_T}  [B, n_T, D]

        # Build additive conditioning tensor aligned with x_tokens layout:
        #   [X_R | X_P | x_t]   lengths: [n_R | n_P | n_T]
        cond = torch.zeros_like(x_tokens)
        cond[:, n_R:n_R + n_s_P, :]              += cond_P   # inject into x_P slice
        cond[:, n_R + n_P:n_R + n_P + n_s_T, :] += cond_T   # inject into x_t slice

        for i, block in enumerate(self.main_blocks):
            scene_c = cond if i == 0 else None
            x_tokens = block(x_tokens, text_tokens, scene_cond=scene_c)

        return x_tokens, scene_out
