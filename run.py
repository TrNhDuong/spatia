"""
run.py
─────────────────────────────────────────────────────────────────────────
Spatia — Train from scratch trên RealEstate10K test set, 1 epoch.

Usage:
    python run.py                          # chạy đầy đủ
    python run.py --max_videos 10          # thử nhanh với 10 videos
    python run.py --skip_download          # đã có video rồi
    python run.py --skip_preprocess        # đã có .pt files rồi
    python run.py --device cpu             # không có GPU
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from configs.config import SpatiaConfig
from models import Spatia
from data.dataset import SpatiaDataset
from training.trainer import train_one_epoch
from utils.checkpoint import save_checkpoint
from pipeline.download import download_metadata, download_videos
from pipeline.preprocess import preprocess_all


# ─────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Spatia: train from scratch on RealEstate10K test set (1 epoch)"
    )
    p.add_argument("--raw_dir",    default="data/raw/realestate",
                   help="Thư mục chứa video raw + metadata")
    p.add_argument("--proc_dir",   default="data/processed_test",
                   help="Thư mục chứa .pt files đã preprocess")
    p.add_argument("--max_videos", type=int, default=100,
                   help="Số video tải về (paper dùng 100 cho test eval)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--device",     default="cuda"
                   if torch.cuda.is_available() else "cpu")
    p.add_argument("--vae_name",   default="Wan-AI/Wan2.2-T2V-1.3B")
    p.add_argument("--t5_name",    default="google/t5-v1_1-xxl")
    p.add_argument("--height",     type=int, default=480)
    p.add_argument("--width",      type=int, default=640)
    p.add_argument("--skip_download",   action="store_true")
    p.add_argument("--skip_preprocess", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Training (Step 4)
# ─────────────────────────────────────────────────────────────────────────
def run_training(proc_dir: Path, cfg: SpatiaConfig, device_str: str):
    device = torch.device(device_str)

    # Dataset
    dataset    = SpatiaDataset(cfg, processed_dir=str(proc_dir))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    n_batches = len(dataloader)
    print(f"\n[Train] {len(dataset)} samples | {n_batches} batches/epoch")

    # Model
    model   = Spatia(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Train] Model: {n_param:.1f} M params")

    # Điều chỉnh iters cho phù hợp với dataset nhỏ
    s1 = min(cfg.stage1_iters, n_batches)
    s2 = min(cfg.stage2_iters, n_batches)

    # ── Stage 1: ControlNet only ──────────────────────────────────────
    print("\n" + "─" * 52)
    print("  Stage 1 — ControlNet blocks (main blocks frozen)")
    print(f"  LR={cfg.lr_controlnet} | iters={s1}")
    print("─" * 52)

    model.freeze_main_blocks()
    opt1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_controlnet, weight_decay=cfg.weight_decay,
    )
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=s1, eta_min=cfg.lr_controlnet * 0.1)

    loss1 = train_one_epoch(model, dataloader, opt1, cfg,
                             device, stage=1, max_iters=s1)
    sch1.step()
    save_checkpoint(model, opt1, 1, loss1, cfg.save_dir, "spatia_stage1")

    # ── Stage 2: LoRA fine-tune main blocks ───────────────────────────
    print("\n" + "─" * 52)
    print("  Stage 2 — LoRA fine-tune main blocks (rank=64)")
    print(f"  LR={cfg.lr_lora} | iters={s2}")
    print("─" * 52)

    model.enable_lora()
    model = model.to(device)
    model.freeze_controlnet()
    model.unfreeze_main_blocks()

    opt2 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_lora, weight_decay=cfg.weight_decay,
    )
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=s2, eta_min=cfg.lr_lora * 0.1)

    loss2 = train_one_epoch(model, dataloader, opt2, cfg,
                             device, stage=2, max_iters=s2)
    sch2.step()
    save_checkpoint(model, opt2, 2, loss2, cfg.save_dir, "spatia_final")

    print(f"\n✓ 1 epoch complete!")
    print(f"  Stage 1 loss : {loss1:.6f}")
    print(f"  Stage 2 loss : {loss2:.6f}")
    print(f"  Checkpoints  : {cfg.save_dir}/")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    raw_dir  = Path(args.raw_dir)
    proc_dir = Path(args.proc_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Config
    cfg             = SpatiaConfig()
    cfg.batch_size  = args.batch_size
    cfg.height      = args.height
    cfg.width       = args.width
    cfg.save_dir    = "checkpoints"

    print("=" * 52)
    print("  Spatia — Test Set Training (1 epoch)")
    print("=" * 52)
    print(f"  Videos     : {args.max_videos}")
    print(f"  Device     : {args.device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Proc dir   : {proc_dir}")
    print("=" * 52)

    # ── Step 1 & 2: Download ──────────────────────────────────────────
    video_list = []
    if not args.skip_download:
        meta_dir   = download_metadata(raw_dir)
        video_dir  = raw_dir / "videos"
        video_list = download_videos(meta_dir, video_dir,
                                     args.max_videos, args.height)
    else:
        print("[Step 1-2] Skipped.")
        video_dir = raw_dir / "videos"
        meta_dir  = raw_dir / "test"
        for vp in sorted(video_dir.glob("*.mp4")):
            tp = meta_dir / (vp.stem + ".txt")
            video_list.append((vp, tp))

    # ── Step 3: Preprocess ────────────────────────────────────────────
    if not args.skip_preprocess:
        preprocess_all(
            video_list, proc_dir, cfg, args.device,
            args.vae_name, args.t5_name,
        )
    else:
        print("[Step 3] Skipped.")

    # ── Step 4: Train ─────────────────────────────────────────────────
    run_training(proc_dir, cfg, args.device)


if __name__ == "__main__":
    main()
