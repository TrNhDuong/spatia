from __future__ import annotations

import argparse
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {v!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Spatia Wan2.2 training pipeline converted from the notebook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Main folders
    p.add_argument("--data-root", type=str, default=None, help="Dataset root containing videos/poses or manifest.csv.")
    p.add_argument("--work-dir", type=str, default=None, help="Output/cache/checkpoint root.")
    p.add_argument("--assets-dir", type=str, default=None, help="Folder produced by download_assets.py.")

    # Asset paths
    p.add_argument("--wan-model", type=str, default=None, help="Local Diffusers Wan2.2 model directory containing model_index.json.")
    p.add_argument("--referdino-repo", type=str, default=None, help="Local ReferDINO repository root.")
    p.add_argument("--referdino-ckpt", type=str, default=None, help="Local ReferDINO checkpoint, e.g. ryt_mevis_swinb.pth.")
    p.add_argument("--mapanything-repo", type=str, default=None, help="Local map-anything repository root.")
    p.add_argument("--mapanything-model", type=str, default=None, help="Local MapAnything model/checkpoint directory.")
    p.add_argument("--keye-model", type=str, default=None, help="Local Keye-VL model directory.")

    # Dataset/model size
    p.add_argument("--train-videos", type=int, default=100)
    p.add_argument("--test-videos", type=int, default=20)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--prev-frames", type=int, default=9)
    p.add_argument("--target-frames", type=int, default=49)
    p.add_argument("--candidate-frames", type=int, default=16)
    p.add_argument("--ref-frames", type=int, default=7)

    # Training args
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--stage", choices=["stage1", "stage2", "both", "none"], default="both")
    p.add_argument("--stage1-steps", "--max-train-steps-stage1", type=int, default=800, dest="max_train_steps_stage1")
    p.add_argument("--stage2-steps", "--max-train-steps-stage2", type=int, default=500, dest="max_train_steps_stage2")
    p.add_argument("--lr-stage1", type=float, default=5e-6)
    p.add_argument("--lr-stage2", type=float, default=1e-6)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--keep-last-ckpts", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Module switches
    p.add_argument("--run-keye", type=str2bool, default=False)
    p.add_argument("--run-referdino", type=str2bool, default=True)
    p.add_argument("--run-mapanything", type=str2bool, default=True)
    p.add_argument("--strict-external-models", type=str2bool, default=True)
    p.add_argument("--preprocess", type=str2bool, default=True)
    p.add_argument("--preprocess-only", action="store_true")
    p.add_argument("--offline", action="store_true", help="Set TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE=1.")

    # Optional extras
    p.add_argument("--run-legacy-kaggle-fixes", action="store_true", help="Run hardcoded Kaggle fix cells from the original notebook.")
    p.add_argument("--setup-lpips-offline", action="store_true", help="Run notebook cell that installs lpips from a Kaggle wheelhouse.")
    p.add_argument("--run-benchmark", action="store_true")
    p.add_argument("--save-sample", action="store_true")
    p.add_argument("--package-output", action="store_true")
    p.add_argument("--audit", action="store_true")

    return p


def parse_args(argv=None):
    args = build_parser().parse_args(argv)

    # Convenience: infer path layout created by download_assets.py.
    if args.assets_dir:
        base = Path(args.assets_dir)
        if args.wan_model is None:
            cand = base / "models" / "wan2.2-ti2v-5b-diffusers"
            if cand.exists():
                args.wan_model = str(cand)
        if args.referdino_repo is None:
            cand = base / "repos" / "ReferDINO"
            if cand.exists():
                args.referdino_repo = str(cand)
        if args.referdino_ckpt is None:
            cand = base / "checkpoints" / "referdino" / "ryt_mevis_swinb.pth"
            if cand.exists():
                args.referdino_ckpt = str(cand)
        if args.mapanything_repo is None:
            cand = base / "repos" / "map-anything"
            if cand.exists():
                args.mapanything_repo = str(cand)
        if args.mapanything_model is None:
            cand = base / "models" / "map-anything"
            if cand.exists():
                args.mapanything_model = str(cand)
        if args.keye_model is None:
            cand = base / "models" / "keye-vl-1.5-8b"
            if cand.exists():
                args.keye_model = str(cand)
    return args
