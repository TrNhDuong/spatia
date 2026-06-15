import torch
import torch.nn as nn
from configs.config import SpatiaConfig
from .blocks import MainBlock, SpatiaNetworkBlock


class Spatia(nn.Module):
    """
    Full Spatia model.

    Inputs (all as latent tokens from Wan2.2 VAE encoder):
        x_t      [B, N_T, C]  – noisy target video tokens (x_0→x_T interpolation)
        x_P      [B, N_P, C]  – preceding video clip tokens
        x_R      [B, N_R, C]  – reference frame tokens (K frames concatenated)
        x_S_T    [B, N_T, C]  – target   scene point-cloud projection tokens
        x_S_P    [B, N_P, C]  – preceding scene point-cloud projection tokens
        text_tokens [B, N_txt, text_dim] – T5 text embeddings
        t        [B]           – timestep ∈ (0, 1)

    Output:
        velocity [B, N_T, C]  – predicted dx_t/dt for Flow Matching loss
    """

    def __init__(self, cfg: SpatiaConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim

        # Input projections: VAE latent C → model dim
        self.video_proj = nn.Linear(cfg.video_latent_dim, dim)
        self.scene_proj = nn.Linear(cfg.video_latent_dim, dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        # 8 Spatia network blocks
        self.blocks = nn.ModuleList([
            SpatiaNetworkBlock(
                dim=dim,
                num_heads=cfg.num_heads,
                text_dim=cfg.text_dim,
                mlp_ratio=cfg.mlp_ratio,
                num_sub_blocks=cfg.num_sub_blocks,
                use_lora=False,
                lora_rank=cfg.lora_rank,
            )
            for _ in range(cfg.num_main_blocks)
        ])

        # Output head: dim → VAE latent C (predict velocity)
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, cfg.video_latent_dim)

    # ──────────────────────────────────────────────
    # Stage helpers
    # ──────────────────────────────────────────────
    def enable_lora(self):
        """
        Stage 2: swap plain Linears in main blocks for LoRA variants.
        Called after Stage-1 training completes.
        """
        from .lora import LoRALinear

        def copy_linear_into_lora(dst: nn.Module, src: nn.Module) -> None:
            if not isinstance(dst, LoRALinear) or not isinstance(src, nn.Linear):
                return
            dst.linear.weight.data.copy_(src.weight.data)
            if src.bias is not None and dst.linear.bias is not None:
                dst.linear.bias.data.copy_(src.bias.data)

        cfg = self.cfg
        for net_block in self.blocks:
            for idx, block in enumerate(net_block.main_blocks):
                new_block = MainBlock(
                    cfg.dim, cfg.num_heads, cfg.text_dim, cfg.mlp_ratio,
                    use_lora=True, lora_rank=cfg.lora_rank,
                )
                new_block.load_state_dict(block.state_dict(), strict=False)
                for name in ("q", "k", "v", "out"):
                    copy_linear_into_lora(
                        getattr(new_block.self_attn, name),
                        getattr(block.self_attn, name),
                    )
                    copy_linear_into_lora(
                        getattr(new_block.cross_attn, name),
                        getattr(block.cross_attn, name),
                    )
                copy_linear_into_lora(new_block.ffn.fc1, block.ffn.fc1)
                copy_linear_into_lora(new_block.ffn.fc2, block.ffn.fc2)
                net_block.main_blocks[idx] = new_block

    def freeze_controlnet(self):
        for net_block in self.blocks:
            for p in net_block.controlnet.parameters():
                p.requires_grad_(False)

    def freeze_main_blocks(self):
        for net_block in self.blocks:
            for p in net_block.main_blocks.parameters():
                p.requires_grad_(False)

    def unfreeze_main_blocks(self):
        """
        Stage 2: enable gradients on main block params.
        If LoRA is enabled, only unfreezes lora_A / lora_B adapters;
        the frozen base linear weight (LoRALinear.linear) is left as-is.
        """
        from .lora import LoRALinear
        for net_block in self.blocks:
            for name, module in net_block.main_blocks.named_modules():
                if isinstance(module, LoRALinear):
                    # Only adapter weights — base weight stays frozen
                    module.lora_A.weight.requires_grad_(True)
                    module.lora_B.weight.requires_grad_(True)
                    # linear.weight stays requires_grad=False (frozen)
                elif isinstance(module, torch.nn.Linear):
                    module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)
                elif isinstance(module, torch.nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad_(True)

    # ──────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────
    def forward(
        self,
        x_t:         torch.Tensor,
        x_P:         torch.Tensor,
        x_R:         torch.Tensor,
        x_S_T:       torch.Tensor,
        x_S_P:       torch.Tensor,
        text_tokens: torch.Tensor,
        t:           torch.Tensor,
    ) -> torch.Tensor:

        B = x_t.shape[0]
        n_R, n_P, n_T = x_R.shape[1], x_P.shape[1], x_t.shape[1]

        # Project inputs to model dim
        x_t   = self.video_proj(x_t)
        x_P   = self.video_proj(x_P)
        x_R   = self.video_proj(x_R)
        x_S_T = self.scene_proj(x_S_T)
        x_S_P = self.scene_proj(x_S_P)

        # Add timestep embedding to noisy target tokens
        t_emb = self.time_embed(t.float().unsqueeze(-1))   # [B, dim]
        x_t   = x_t + t_emb.unsqueeze(1)

        # Concatenate token sequences
        x_tokens     = torch.cat([x_R, x_P, x_t],   dim=1)  # [B, N_R+N_P+N_T, D]
        scene_tokens = torch.cat([x_S_P, x_S_T],    dim=1)  # [B, N_P+N_T,     D]

        for block in self.blocks:
            x_tokens, scene_tokens = block(
                x_tokens, scene_tokens, text_tokens, n_R, n_P, n_T
            )

        # Extract only the target part (last N_T tokens)
        x_out    = x_tokens[:, n_R + n_P:, :]
        velocity = self.out_proj(self.out_norm(x_out))   # [B, N_T, C]
        return velocity
