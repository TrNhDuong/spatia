"""
spatia_pipeline/__init__.py
-----------------------------
Public API for the spatia_pipeline package.
"""

from spatia_pipeline.config import SpatiaConfig
from spatia_pipeline.data.dataset import SpatiaFullDataset, build_dataloaders
from spatia_pipeline.evaluation.benchmark import run_benchmark
from spatia_pipeline.evaluation.metrics import (
    compute_psnr,
    compute_ssim_video,
    compute_lpips_video_01,
)
from spatia_pipeline.model.control_net import LatentSpatiaControlNet
from spatia_pipeline.model.wan_trainer import WanSpatiaTrainer
from spatia_pipeline.training.trainer import (
    train_loop,
    set_stage1_trainable,
    set_stage2_trainable,
    evaluate,
)

__version__ = "2.0.0"

__all__ = [
    "SpatiaConfig",
    "WanSpatiaTrainer",
    "LatentSpatiaControlNet",
    "SpatiaFullDataset",
    "build_dataloaders",
    "train_loop",
    "set_stage1_trainable",
    "set_stage2_trainable",
    "evaluate",
    "run_benchmark",
    "compute_psnr",
    "compute_ssim_video",
    "compute_lpips_video_01",
]
