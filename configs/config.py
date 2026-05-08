from dataclasses import dataclass


@dataclass
class SpatiaConfig:
    # ── Model ──────────────────────────────────────────
    dim: int = 1024             # hidden dim (full paper: ~5B params)
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_main_blocks: int = 8    # 8 network blocks total
    num_sub_blocks: int = 4     # 4 main DiT blocks per network block
    text_dim: int = 4096        # T5 text encoder output dim
    video_latent_dim: int = 16  # Wan2.2 VAE latent channels
    max_ref_frames: int = 7     # K=7 reference frames (paper default)

    # ── Training ───────────────────────────────────────
    batch_size: int = 2
    num_epochs: int = 1
    lr_controlnet: float = 1e-5   # Stage 1 learning rate
    lr_lora: float = 1e-4         # Stage 2 learning rate
    stage1_iters: int = 8000      # ControlNet-only training iterations
    stage2_iters: int = 5000      # LoRA fine-tuning iterations
    lora_rank: int = 64
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    aug_t_max: float = 50.0 / 1000  # t_aug ∈ [0, 50] normalized to [0, 1]

    # ── Video / Latent dims ────────────────────────────
    target_frames: int = 81     # frames for first (image-conditioned) iteration
    preceding_frames: int = 9   # frames used as preceding context
    height: int = 480
    width: int = 640
    spatial_downsample: int = 16
    temporal_downsample: int = 4

    # ── Checkpointing ──────────────────────────────────
    save_dir: str = "checkpoints"
    log_every: int = 50
