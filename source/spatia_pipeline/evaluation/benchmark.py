"""
spatia_pipeline/evaluation/benchmark.py
-----------------------------------------
Benchmark evaluation producing PSNR / SSIM / LPIPS / WorldScore-proxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from spatia_pipeline.config import SpatiaConfig
from spatia_pipeline.evaluation.metrics import (
    compute_lpips_video_01,
    compute_psnr,
    compute_ssim_video,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.detach().float().clamp(-1, 1) + 1) / 2).clamp(0, 1)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_benchmark(
    model,
    loader: DataLoader,
    cfg: SpatiaConfig,
    max_batches: int = 20,
) -> Tuple[pd.DataFrame, dict]:
    """
    Evaluate ``model`` on ``loader`` for up to ``max_batches`` batches.

    Computes per-clip:
        psnr, ssim, lpips          — over all frames
        psnr_c, ssim_c, lpips_c   — last (camera-control) frame only

    Returns:
        df:      per-clip DataFrame
        summary: mean metrics + WorldScore proxy breakdown
    """
    model.eval()
    rows = []

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        pred, target, batch = model.reconstruct_video(batch, t_value=0.2)
        pred01   = _to_01(pred)
        target01 = _to_01(target)

        for b in range(pred01.shape[0]):
            pv = pred01[b]
            tv = target01[b]
            row = {
                "id":     batch["id"][b] if isinstance(batch.get("id"), list) else str(bi),
                "psnr":   compute_psnr(pv, tv),
                "ssim":   compute_ssim_video(pv, tv),
                "lpips":  compute_lpips_video_01(pv, tv, cfg.device),
                "psnr_c": compute_psnr(pv[-1], tv[-1]),
                "ssim_c": _ssim_frame(pv[-1], tv[-1]),
                "lpips_c": compute_lpips_video_01(pv[-1:], tv[-1:], cfg.device),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    summary: dict = {}

    if len(df) > 0:
        for col in ["psnr", "ssim", "lpips", "psnr_c", "ssim_c", "lpips_c"]:
            summary[col] = float(df[col].mean())

        static_score  = float(np.clip(summary["ssim"] * 100, 0, 100))
        dynamic_score = float(np.clip((1 - summary["lpips"]) * 100, 0, 100))
        camera_ctrl   = float(np.clip((summary["psnr_c"] / 30) * 100, 0, 100))
        avg_score     = float(np.mean([static_score, dynamic_score, camera_ctrl]))

        summary.update({
            "worldscore_avg_proxy":         avg_score,
            "worldscore_static_proxy":      static_score,
            "worldscore_dynamic_proxy":     dynamic_score,
            "worldscore_camera_ctrl_proxy": camera_ctrl,
            "lpips_source":                 "lpips_alex",
            "num_eval_samples":             int(len(df)),
            "benchmark_note": (
                "WorldScore values are proxy only; PSNR/SSIM/LPIPS are computed "
                "from reconstruction on validation samples."
            ),
        })

    return df, summary


def _ssim_frame(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Single-frame SSIM ([C, H, W] in [0, 1])."""
    from spatia_pipeline.evaluation.metrics import _ssim_single  # type: ignore[attr-defined]
    return _ssim_single(pred, target)


# ---------------------------------------------------------------------------
# Save benchmark results
# ---------------------------------------------------------------------------

def save_benchmark_results(
    df: pd.DataFrame,
    summary: dict,
    ckpt_dir: Path,
) -> None:
    if len(df) == 0:
        return
    df.to_csv(ckpt_dir / "benchmark_results.csv", index=False)
    with open(ckpt_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved benchmark CSV + JSON to", ckpt_dir)
