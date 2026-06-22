"""
spatia_pipeline/model/control_net.py
--------------------------------------
Spatia-style latent-space control branch.

  SpatioTemporalResidualBlock  — single residual block with temporal + spatial Conv3D
  LatentSpatiaControlNet       — full control branch (ControlNet-style residual output)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalResidualBlock(nn.Module):
    """
    Residual block with separate temporal and spatial 3-D convolutions
    followed by a channel-MLP (FFN).
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1    = nn.GroupNorm(32, width)
        self.temporal = nn.Conv3d(width, width, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.norm2   = nn.GroupNorm(32, width)
        self.spatial = nn.Conv3d(width, width, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.norm3 = nn.GroupNorm(32, width)
        self.ffn   = nn.Sequential(
            nn.Conv3d(width, width * 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(width * 4, width, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal(F.silu(self.norm1(x)))
        x = x + self.spatial(F.silu(self.norm2(x)))
        x = x + self.ffn(F.silu(self.norm3(x)))
        return x


class LatentSpatiaControlNet(nn.Module):
    """
    Stable latent-space Spatia-style control branch.

    Inputs:
        noisy_latents  — [B, C, T, H, W] noisy Wan latent
        cond_latents   — [B, C', T, H, W] encoded condition (prev/control/memory/ref)
        t              — scalar or [B] flow timestep in [0, 1]

    Output:
        residual — [B, C, T, H, W]  (same shape as Wan transformer velocity prediction)

    The output projection is zero-initialised so training starts from the
    unmodified Wan baseline (ControlNet-style safe start).
    """

    def __init__(
        self,
        channels: int,
        width: int = 384,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width    = width
        self.depth    = depth

        self.input_proj  = nn.Conv3d(channels * 2 + 1, width, kernel_size=3, padding=1)
        self.blocks      = nn.ModuleList([SpatioTemporalResidualBlock(width) for _ in range(depth)])
        self.output_norm = nn.GroupNorm(32, width)
        self.output_proj = nn.Conv3d(width, channels, kernel_size=3, padding=1)

        # Zero-init: start from identity (no control residual)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)  # type: ignore[arg-type]

    def forward(
        self,
        noisy_latents: torch.Tensor,
        cond_latents: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Align spatial / temporal resolution
        if cond_latents.shape[-3:] != noisy_latents.shape[-3:]:
            cond_latents = F.interpolate(
                cond_latents,
                size=noisy_latents.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        # Align channels
        if cond_latents.shape[1] != noisy_latents.shape[1]:
            c = noisy_latents.shape[1]
            if cond_latents.shape[1] < c:
                rep = math.ceil(c / cond_latents.shape[1])
                cond_latents = cond_latents.repeat(1, rep, 1, 1, 1)[:, :c]
            else:
                cond_latents = cond_latents[:, :c]

        # Timestep embedding map
        if t.ndim == 0:
            t = t[None]
        t_map = t.float().view(-1, 1, 1, 1, 1)
        t_map = t_map.to(dtype=noisy_latents.dtype, device=noisy_latents.device)
        t_map = t_map.expand(
            noisy_latents.shape[0], 1,
            noisy_latents.shape[2],
            noisy_latents.shape[3],
            noisy_latents.shape[4],
        )

        x = torch.cat([noisy_latents, cond_latents, t_map], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(F.silu(self.output_norm(x)))
        return x
