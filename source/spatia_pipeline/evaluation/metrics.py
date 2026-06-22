"""
spatia_pipeline/evaluation/metrics.py
----------------------------------------
Video quality metrics: PSNR, SSIM, LPIPS, and inference sample saving.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from spatia_pipeline.config import SpatiaConfig


# ---------------------------------------------------------------------------
# LPIPS (lazy-loaded)
# ---------------------------------------------------------------------------

_LPIPS_FN = None


def _get_lpips_fn(device: torch.device, net: str = "alex"):
    global _LPIPS_FN
    if _LPIPS_FN is None:
        import lpips as _lpips  # type: ignore
        _LPIPS_FN = _lpips.LPIPS(net=net).to(device).eval()
    return _LPIPS_FN


@torch.no_grad()
def compute_lpips_video_01(
    pred01: torch.Tensor,
    target01: torch.Tensor,
    device: torch.device,
    batch_frames: int = 4,
) -> float:
    """
    Compute mean LPIPS over all frames.

    Args:
        pred01/target01: [T, C, H, W] or [B, T, C, H, W], range [0, 1]
        device: torch device
        batch_frames: frames per LPIPS forward pass (to limit VRAM)
    """
    if pred01.ndim == 5:
        pred01   = pred01.flatten(0, 1)
        target01 = target01.flatten(0, 1)

    pred   = pred01.float().clamp(0, 1) * 2 - 1
    target = target01.float().clamp(0, 1) * 2 - 1

    fn = _get_lpips_fn(device)
    vals = []
    for i in range(0, pred.shape[0], batch_frames):
        p = pred[i:i + batch_frames].to(device)
        t = target[i:i + batch_frames].to(device)
        vals.append(fn(p, t).view(-1).detach().cpu())
    return float(torch.cat(vals).mean().item())


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR in dB. Both tensors in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    return -10 * math.log10(max(mse, 1e-8))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _ssim_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM for a single [C, H, W] frame in [0, 1]."""
    try:
        from skimage.metrics import structural_similarity as _ssim  # type: ignore
        p = pred.detach().cpu().permute(1, 2, 0).numpy()
        t = target.detach().cpu().permute(1, 2, 0).numpy()
        return float(_ssim(t, p, channel_axis=2, data_range=1.0))
    except ImportError:
        pass

    # Pure-PyTorch fallback
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = pred.mean();  mu_y = target.mean()
    var_x = pred.var(unbiased=False)
    var_y = target.var(unbiased=False)
    cov   = ((pred - mu_x) * (target - mu_y)).mean()
    num   = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
    den   = (mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)
    return float((num / den).detach().cpu())


def compute_ssim_video(pred: torch.Tensor, target: torch.Tensor, max_frames: int = 8) -> float:
    """SSIM averaged over up to ``max_frames`` uniformly sampled frames. Both [T, C, H, W]."""
    T    = min(pred.shape[0], max_frames)
    idxs = torch.linspace(0, pred.shape[0] - 1, steps=T).long().tolist()
    return float(np.mean([_ssim_single(pred[i], target[i]) for i in idxs]))


# ---------------------------------------------------------------------------
# Tensor → video conversion + sample saving
# ---------------------------------------------------------------------------

def tensor_to_uint8_video(x: torch.Tensor) -> np.ndarray:
    """[T, C, H, W] tensor in [-1, 1] → uint8 [T, H, W, 3] numpy array."""
    x = ((x.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy()
    return np.transpose(x, (0, 2, 3, 1))


@torch.no_grad()
def save_proxy_sample(
    model,
    loader,
    out_path: Path | str,
    cfg: SpatiaConfig,
    t_value: float = 0.2,
) -> Path:
    """
    Reconstruct one batch from ``loader`` and save it as an mp4 video.
    """
    model.eval()
    batch = next(iter(loader))
    pred, _target, _ = model.reconstruct_video(batch, t_value=t_value)
    vid = tensor_to_uint8_video(pred[0])

    out_path = Path(out_path)
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, 8, (vid.shape[2], vid.shape[1]))
    for fr in vid:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
    model.train()
    print("Saved sample:", out_path)
    return out_path
