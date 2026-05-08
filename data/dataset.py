import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
from configs.config import SpatiaConfig


class SpatiaDataset(Dataset):
    """
    Spatia training dataset.

    Hai chế độ hoạt động:
    ─────────────────────────────────────────────────────────
    1. REAL MODE  (processed_dir có file .pt)
       Tải pre-processed latent tensors từ disk.
       Các file .pt được tạo bởi scripts/preprocess.py.

    2. DUMMY MODE  (processed_dir trống hoặc None)
       Trả về random tensors để test kiến trúc mà không cần data.
    ─────────────────────────────────────────────────────────

    Mỗi file .pt là một dict:
        x_T    [N_T, C]          target video latent
        x_P    [N_P, C]          preceding video latent
        x_R    [N_R, C]          K reference frame latents (stacked)
        x_S_T  [N_T, C]          target scene projection latent
        x_S_P  [N_P, C]          preceding scene projection latent
        text   [N_txt, text_dim] T5 text embedding
    """

    def __init__(self, cfg: SpatiaConfig,
                 processed_dir: str | None = None,
                 dummy_samples: int = 500):
        self.cfg = cfg
        self._compute_token_counts()

        # ── Detect real data ──────────────────────────────────────────
        self.real_files: list[Path] = []
        if processed_dir and Path(processed_dir).exists():
            self.real_files = sorted(Path(processed_dir).glob("*.pt"))

        self.is_real = len(self.real_files) > 0
        if self.is_real:
            print(f"[SpatiaDataset] REAL MODE — {len(self.real_files)} samples "
                  f"from '{processed_dir}'")
        else:
            self._dummy_n = dummy_samples
            print(f"[SpatiaDataset] DUMMY MODE — {dummy_samples} random samples")

    # ── Internal helpers ──────────────────────────────────────────────
    def _compute_token_counts(self):
        cfg = self.cfg
        h   = cfg.height  // cfg.spatial_downsample
        w   = cfg.width   // cfg.spatial_downsample
        t_T = cfg.target_frames    // cfg.temporal_downsample
        t_P = cfg.preceding_frames // cfg.temporal_downsample
        self.N_T   = t_T * h * w
        self.N_P   = t_P * h * w
        self.N_R   = cfg.max_ref_frames * h * w
        self.N_txt = 77

    def _dummy_item(self) -> dict:
        C = self.cfg.video_latent_dim
        D = self.cfg.text_dim
        return {
            "x_T":   torch.randn(self.N_T,   C),
            "x_P":   torch.randn(self.N_P,   C),
            "x_R":   torch.randn(self.N_R,   C),
            "x_S_T": torch.randn(self.N_T,   C),
            "x_S_P": torch.randn(self.N_P,   C),
            "text":  torch.randn(self.N_txt, self.cfg.text_dim),
        }

    def _pad_or_crop(self, t: torch.Tensor, target_len: int) -> torch.Tensor:
        """Ensure first dim == target_len (pad zeros or truncate)."""
        n = t.shape[0]
        if n == target_len:
            return t
        if n > target_len:
            return t[:target_len]
        # Pad
        pad = torch.zeros(target_len - n, *t.shape[1:], dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    # ── Dataset interface ─────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.real_files) if self.is_real else self._dummy_n

    def __getitem__(self, idx: int) -> dict:
        if not self.is_real:
            return self._dummy_item()

        data = torch.load(self.real_files[idx], map_location="cpu",
                          weights_only=True)

        # Ensure consistent shapes (in case pre-processing used different cfg)
        return {
            "x_T":   self._pad_or_crop(data["x_T"],   self.N_T),
            "x_P":   self._pad_or_crop(data["x_P"],   self.N_P),
            "x_R":   self._pad_or_crop(data["x_R"],   self.N_R),
            "x_S_T": self._pad_or_crop(data["x_S_T"], self.N_T),
            "x_S_P": self._pad_or_crop(data["x_S_P"], self.N_P),
            "text":  self._pad_or_crop(data["text"],   self.N_txt),
        }
