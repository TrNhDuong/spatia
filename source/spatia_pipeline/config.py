"""
spatia_pipeline/config.py
--------------------------
Central configuration dataclass.  All notebook-level globals (DEVICE, AMP_DTYPE,
LORA_RANK, ...) are consolidated here so every module can receive a single
``SpatiaConfig`` instance instead of reading bare globals.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _default_work_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return Path("/workspace/outputs/spatia_full_work")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpatiaConfig:
    """
    All pipeline hyper-parameters in one place.
    Create via :py:func:`SpatiaConfig.default` or :py:func:`SpatiaConfig.from_args`.
    """

    # ---- Seeds ----
    seed: int = 42

    # ---- Resolution / frames ----
    height: int = 192
    width: int = 320
    prev_frames: int = 9
    target_frames: int = 49
    candidate_frames: int = 16
    ref_frames: int = 7

    # ---- Dataset split ----
    train_videos: int = 100
    test_videos: int = 20

    # ---- Prompts ----
    default_prompt: str = "A realistic real estate video with smooth camera movement."
    use_prompt_csv_if_found: bool = True

    # ---- Module toggles ----
    run_keye: bool = False
    run_referdino: bool = True
    run_mapanything: bool = True
    strict_external_models: bool = True

    # ---- Wan2.2 backbone ----
    use_wan2_backbone: bool = True
    allow_toy_adapter: bool = False
    strict_wan_backbone: bool = True
    enable_gradient_checkpointing: bool = True

    # ---- Training knobs ----
    batch_size: int = 1
    num_workers: Optional[int] = None      # None → auto
    grad_accum_steps: int = 4

    max_train_steps_stage1: int = 800
    max_train_steps_stage2: int = 500
    paper_stage1_steps: int = 8000
    paper_stage2_steps: int = 5000

    lr_stage1: float = 5e-6
    lr_stage2: float = 1e-6

    log_every: int = 25
    val_every: int = 100
    save_every: int = 100
    eval_max_batches: int = 1
    keep_last_ckpts: int = 1
    min_free_gb_for_save: float = 1.0

    # ---- LoRA ----
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # ---- Control branch ----
    control_hidden_mult: int = 2   # legacy; not used by current branch
    control_width: int = 384
    control_depth: int = 6
    control_output_scale: float = 0.50

    # ---- Optimizer ----
    grad_clip_stage1: float = 0.50
    grad_clip_stage2: float = 0.25
    optim_weight_decay: float = 1e-4
    optim_eps: float = 1e-6
    warmup_ratio: float = 0.10
    min_lr_scale: float = 0.10
    max_consecutive_bad_steps: int = 25

    # ---- Flow-matching stability ----
    timestep_min: float = 0.02
    timestep_max: float = 0.98
    latent_clamp_value: float = 8.0
    noise_clamp_value: float = 4.0
    pred_clamp_value: float = 10.0
    loss_diff_clamp_value: float = 5.0

    control_loss_static_weight: float = 1.0
    control_loss_dynamic_weight: float = 0.75

    # ---- Directories (resolved at runtime) ----
    work_dir: Path = field(default_factory=_default_work_dir)
    run_tag: str = ""              # filled by setup_dirs()

    # computed from work_dir + run_tag
    cache_dir: Path = field(default=None)   # type: ignore[assignment]
    proc_dir: Path = field(default=None)    # type: ignore[assignment]
    ckpt_dir: Path = field(default=None)    # type: ignore[assignment]
    sample_dir: Path = field(default=None)  # type: ignore[assignment]

    # ---- Asset paths ----
    data_root: Optional[Path] = None
    wan_model: Optional[Path] = None
    wan_dir: Optional[Path] = None         # resolved diffusers dir
    referdino_repo: Optional[Path] = None
    referdino_input_repo: Optional[Path] = None
    referdino_ckpt: Optional[Path] = None
    mapanything_repo: Optional[Path] = None
    mapanything_model: Optional[Path] = None
    keye_model: Optional[Path] = None

    # ---- Runtime (set by setup_device) ----
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    amp_dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    backbone_dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    use_grad_scaler: bool = False

    # ---------------------------------------------------------------------------

    @property
    def total_sample_frames(self) -> int:
        return self.candidate_frames + self.prev_frames + self.target_frames

    @property
    def max_videos(self) -> int:
        return self.train_videos + self.test_videos

    @property
    def effective_num_workers(self) -> int:
        if self.num_workers is not None:
            return self.num_workers
        return 2 if os.name != "nt" else 0

    # ---------------------------------------------------------------------------
    # Factory methods
    # ---------------------------------------------------------------------------

    @classmethod
    def default(cls) -> "SpatiaConfig":
        cfg = cls()
        cfg.setup_device()
        cfg.setup_dirs()
        return cfg

    @classmethod
    def from_args(cls, args) -> "SpatiaConfig":
        """Build a config from argparse namespace (produced by args.parse_args)."""
        cfg = cls()

        # Basic scalars
        for attr in [
            "seed", "height", "width", "prev_frames", "target_frames",
            "candidate_frames", "ref_frames", "train_videos", "test_videos",
            "batch_size", "grad_accum_steps",
            "max_train_steps_stage1", "max_train_steps_stage2",
            "lr_stage1", "lr_stage2",
            "lora_rank", "lora_alpha", "lora_dropout",
            "log_every", "val_every", "save_every", "keep_last_ckpts",
            "run_keye", "run_referdino", "run_mapanything", "strict_external_models",
        ]:
            arg_attr = attr.replace("_", "-").replace("-", "_")  # normalise dashes
            # argparse replaces dashes with underscores in dest
            dest = attr.replace("-", "_")
            if hasattr(args, dest):
                setattr(cfg, attr, getattr(args, dest))

        if hasattr(args, "num_workers") and args.num_workers is not None:
            cfg.num_workers = int(args.num_workers)

        # Directories
        if getattr(args, "work_dir", None):
            cfg.work_dir = Path(args.work_dir).expanduser().resolve()

        # Asset paths
        path_map = {
            "data_root": "data_root",
            "wan_model": "wan_model",
            "referdino_repo": "referdino_repo",
            "referdino_ckpt": "referdino_ckpt",
            "mapanything_repo": "mapanything_repo",
            "mapanything_model": "mapanything_model",
            "keye_model": "keye_model",
        }
        for cfg_attr, arg_attr in path_map.items():
            val = getattr(args, arg_attr, None)
            if val:
                setattr(cfg, cfg_attr, Path(val).expanduser().resolve())

        cfg.setup_device()
        cfg.setup_dirs()
        return cfg

    # ---------------------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------------------

    def setup_device(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
            torch.backends.cuda.matmul.allow_tf32 = True   # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True          # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = True           # type: ignore[attr-defined]
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.amp_dtype = _pick_amp_dtype(self.device)
        self.backbone_dtype = self.amp_dtype if self.device.type == "cuda" else torch.float32
        self.use_grad_scaler = bool(
            self.device.type == "cuda" and self.amp_dtype == torch.float16
        )
        print("Device:", self.device, "| AMP dtype:", self.amp_dtype,
              "| GradScaler:", self.use_grad_scaler)

    def setup_dirs(self, run_tag: str = "") -> None:
        if run_tag:
            self.run_tag = run_tag
        if not self.run_tag:
            self.run_tag = (
                f"wan2_p{self.prev_frames}_t{self.target_frames}"
                f"_c{self.candidate_frames}_r{self.ref_frames}"
                f"_{self.train_videos}vid_stable"
            )

        self.cache_dir  = self.work_dir / f"spatia_full_cache_{self.run_tag}"
        self.proc_dir   = self.work_dir / f"processed_spatia_full_{self.run_tag}"
        self.ckpt_dir   = self.work_dir / f"spatia_full_checkpoints_{self.run_tag}"
        self.sample_dir = self.work_dir / f"spatia_full_samples_{self.run_tag}"

        for d in [self.cache_dir, self.proc_dir, self.ckpt_dir, self.sample_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print("Work dir:", self.work_dir)
        print("Run tag:", self.run_tag)
        print("Proc dir:", self.proc_dir)

    def lr_scale(self, step: int, max_steps: int) -> float:
        """Cosine LR schedule with linear warmup."""
        warmup_steps = max(1, int(max_steps * self.warmup_ratio))
        if step <= warmup_steps:
            return max(self.min_lr_scale, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine

    def grad_clip(self, stage_name: str) -> float:
        return self.grad_clip_stage2 if "stage2" in stage_name.lower() else self.grad_clip_stage1

    def summary(self) -> None:
        print(f"TOTAL_SAMPLE_FRAMES: {self.total_sample_frames}")
        print(f"Train/Test: {self.train_videos} / {self.test_videos}")
        print(f"Effective batch: {self.batch_size * self.grad_accum_steps}")
        print(f"Wan backbone required: {self.use_wan2_backbone}, strict: {self.strict_wan_backbone}")
