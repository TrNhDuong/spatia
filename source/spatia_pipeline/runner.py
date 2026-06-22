"""
spatia_pipeline/runner.py
---------------------------
Main pipeline runner.  Replaces the old exec()-based runner with direct module imports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch

from spatia_pipeline.args import parse_args
from spatia_pipeline.assets.path_finder import (
    add_repo_paths,
    autodetect_local_paths,
    find_keye_model,
    find_mapanything_model,
    find_mapanything_repo,
    find_patched_referdino_repo,
    find_referdino_ckpt,
    find_wan_model,
    resolve_diffusers_model_dir,
)
from spatia_pipeline.config import SpatiaConfig
from spatia_pipeline.data.dataset import build_dataloaders, split_files
from spatia_pipeline.data.manifest import load_manifest
from spatia_pipeline.evaluation.benchmark import run_benchmark, save_benchmark_results
from spatia_pipeline.evaluation.metrics import save_proxy_sample
from spatia_pipeline.model.wan_trainer import WanSpatiaTrainer, count_module_params
from spatia_pipeline.preprocessing.keye_module import KeyeModule
from spatia_pipeline.preprocessing.mapanything_module import MapAnythingModule
from spatia_pipeline.preprocessing.preprocess import run_preprocess
from spatia_pipeline.preprocessing.referdino_module import ReferDinoModule
from spatia_pipeline.setup.dependencies import check_and_install
from spatia_pipeline.setup.env_setup import create_offline_shims, set_offline_env
from spatia_pipeline.training.checkpoint import safe_torch_save
from spatia_pipeline.training.trainer import (
    set_stage1_trainable,
    set_stage2_trainable,
    train_loop,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: SpatiaConfig, args) -> SpatiaConfig:
    """
    Fill asset paths in cfg:
    1. CLI args take priority
    2. Kaggle /kaggle/input auto-detect
    3. Local assets_dir (from download_assets.py)
    """
    kaggle_input = Path("/kaggle/input")

    # Prefer CLI; fall back to auto-detect
    if cfg.data_root is None:
        from spatia_pipeline.assets.path_finder import (
            find_dir_by_keywords,
            KAGGLE_INPUT,
        )
        cfg.data_root = (
            find_dir_by_keywords(["realestate"], ["*.mp4", "*.txt", "*.csv"], root=KAGGLE_INPUT)
            or find_dir_by_keywords(["real", "estate"], ["*.mp4", "*.txt", "*.csv"], root=KAGGLE_INPUT)
        )

    if cfg.wan_model is None:
        cfg.wan_model = find_wan_model(kaggle_input)

    if cfg.referdino_repo is None:
        cfg.referdino_repo = find_patched_referdino_repo(kaggle_input)
        cfg.referdino_input_repo = cfg.referdino_repo

    if cfg.referdino_ckpt is None:
        cfg.referdino_ckpt = find_referdino_ckpt(kaggle_input)

    if cfg.mapanything_repo is None:
        cfg.mapanything_repo = find_mapanything_repo(kaggle_input)

    if cfg.mapanything_model is None:
        cfg.mapanything_model = find_mapanything_model(kaggle_input)

    if cfg.keye_model is None:
        cfg.keye_model = find_keye_model(kaggle_input)

    # Also try local assets_dir
    if getattr(args, "assets_dir", None):
        local = autodetect_local_paths(Path(args.assets_dir))
        for key, val in local.items():
            if val is not None and getattr(cfg, key, None) is None:
                setattr(cfg, key, val)

    # Resolve Wan diffusers dir
    if cfg.wan_model is not None and cfg.wan_dir is None:
        try:
            cfg.wan_dir = resolve_diffusers_model_dir(cfg.wan_model, "Wan2.2")
        except FileNotFoundError as e:
            print("WARN:", e)

    print("\nResolved asset paths:")
    for name in [
        "DATA_ROOT", "WAN_MODEL", "WAN_DIR",
        "REFERDINO_REPO", "REFERDINO_CKPT",
        "MAPANYTHING_REPO", "MAPANYTHING_MODEL", "KEYE_MODEL",
    ]:
        attr = name.lower()
        print(f"  {name:20s} -> {getattr(cfg, attr, None)}")

    return cfg


# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------

def _final_save(cfg: SpatiaConfig, model, train_ds, benchmark_summary: dict) -> None:
    final_ckpt  = cfg.ckpt_dir / "spatia_wan2_trainable_delta_final.pt"
    config_path = cfg.ckpt_dir / "spatia_wan2_config.json"

    payload = {
        "trainable_state": model.export_trainable_state(),
        "prompt_to_id":    train_ds.prompt_to_id,
        "config": {
            "height":             cfg.height,
            "width":              cfg.width,
            "prev_frames":        cfg.prev_frames,
            "target_frames":      cfg.target_frames,
            "candidate_frames":   cfg.candidate_frames,
            "ref_frames":         cfg.ref_frames,
            "lora_rank":          cfg.lora_rank,
            "lora_alpha":         cfg.lora_alpha,
            "lora_dropout":       cfg.lora_dropout,
            "amp_dtype":          str(cfg.amp_dtype),
            "effective_batch":    cfg.batch_size * cfg.grad_accum_steps,
            "lr_stage1":          cfg.lr_stage1,
            "lr_stage2":          cfg.lr_stage2,
            "grad_clip_stage1":   cfg.grad_clip_stage1,
            "grad_clip_stage2":   cfg.grad_clip_stage2,
            "control_width":      cfg.control_width,
            "control_depth":      cfg.control_depth,
            "control_output_scale": cfg.control_output_scale,
            "wan_dir":            str(cfg.wan_dir),
            "default_prompt":     cfg.default_prompt,
            "benchmark_summary":  benchmark_summary,
            "note": (
                "Wan2.2-backed Spatia-style training. "
                "Saves control branch + Wan LoRA delta only, not full Wan2.2 base weights."
            ),
        },
    }
    safe_torch_save(payload, final_ckpt, min_free_gb=cfg.min_free_gb_for_save)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_videos":       cfg.train_videos,
            "test_videos":        cfg.test_videos,
            "height":             cfg.height,
            "width":              cfg.width,
            "prev_frames":        cfg.prev_frames,
            "target_frames":      cfg.target_frames,
            "candidate_frames":   cfg.candidate_frames,
            "ref_frames":         cfg.ref_frames,
            "stage1_steps":       cfg.max_train_steps_stage1,
            "stage2_steps":       cfg.max_train_steps_stage2,
            "paper_stage1_steps": cfg.paper_stage1_steps,
            "paper_stage2_steps": cfg.paper_stage2_steps,
            "lora_rank":          cfg.lora_rank,
            "lora_alpha":         cfg.lora_alpha,
            "lora_dropout":       cfg.lora_dropout,
            "amp_dtype":          str(cfg.amp_dtype),
            "effective_batch":    cfg.batch_size * cfg.grad_accum_steps,
            "lr_stage1":          cfg.lr_stage1,
            "lr_stage2":          cfg.lr_stage2,
            "grad_clip_stage1":   cfg.grad_clip_stage1,
            "grad_clip_stage2":   cfg.grad_clip_stage2,
            "benchmark_summary":  benchmark_summary,
            "pipeline_inputs": {
                "data_root":        str(cfg.data_root),
                "wan_model":        str(cfg.wan_model),
                "wan_dir":          str(cfg.wan_dir),
                "mapanything_model":str(cfg.mapanything_model),
                "keye_model":       str(cfg.keye_model),
                "referdino_ckpt":   str(cfg.referdino_ckpt),
                "mapanything_repo": str(cfg.mapanything_repo),
                "referdino_repo":   str(cfg.referdino_repo),
            },
        }, f, indent=2)

    print("Saved final trainable delta:", final_ckpt)
    print("Saved config:", config_path)


# ---------------------------------------------------------------------------
# Paper audit
# ---------------------------------------------------------------------------

def _paper_audit(cfg: SpatiaConfig, model) -> None:
    print("===== PAPER COMPLIANCE AUDIT =====")
    print("Training model class:", type(model).__name__)
    print("Wan pipeline class:", type(model.pipe).__name__)
    print("Transformer class:", type(model.transformer).__name__)
    print("VAE class:", type(model.vae).__name__)
    print("Text encoder class:", type(model.text_encoder).__name__)

    wan_params = (
        count_module_params(model.transformer)
        + count_module_params(model.vae)
        + count_module_params(model.text_encoder)
    )
    control_params = 0 if model.control is None else count_module_params(model.control)
    lora_params = sum(
        p.numel()
        for n, p in model.transformer.named_parameters()
        if "lora" in n.lower()
    )
    print(f"Wan component params: {wan_params / 1e9:.3f}B")
    print(f"Control params: {control_params}")
    print(f"LoRA params: {lora_params}")
    print(f"LoRA rank: {cfg.lora_rank}")
    print(f"AMP dtype: {cfg.amp_dtype}")
    print(f"Stage LR: {cfg.lr_stage1} / {cfg.lr_stage2}")
    print(f"Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")
    print(f"Frames: prev/target/candidate/ref = {cfg.prev_frames}/{cfg.target_frames}/{cfg.candidate_frames}/{cfg.ref_frames}")
    print(f"Resolution: {cfg.height}x{cfg.width}")

    if type(model).__name__ != "WanSpatiaTrainer":
        raise RuntimeError("Wrong model class.")
    if wan_params < 1_000_000_000:
        raise RuntimeError("Wan backbone parameter audit failed.")
    if cfg.lora_rank != 64:
        raise RuntimeError("LoRA rank is not paper-aligned rank 64.")
    if cfg.prev_frames != 9:
        raise RuntimeError("PREV_FRAMES is not 9.")
    if cfg.ref_frames != 7:
        raise RuntimeError("REF_FRAMES is not 7.")

    print("Audit OK: notebook is using Wan2.2-backed training.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(args=None) -> dict:
    """
    Run the full Spatia training pipeline.

    Args:
        args: parsed argparse namespace, or None to parse from sys.argv.

    Returns:
        state dict with 'cfg', 'model', 'train_ds', 'val_ds', 'benchmark_summary'.
    """
    if args is None:
        args = parse_args()

    # ---------- Config ----------
    cfg = SpatiaConfig.from_args(args)
    cfg.summary()

    # ---------- Offline / deps ----------
    if getattr(args, "offline", False):
        set_offline_env()

    check_and_install()

    # Create offline shims (for Kaggle environments)
    offline_pkgs_dir = cfg.work_dir / "offline_pkgs"
    create_offline_shims(offline_pkgs_dir)

    # ---------- Path detection ----------
    cfg = _resolve_paths(cfg, args)

    if cfg.data_root is None:
        raise FileNotFoundError("Cannot find dataset. Pass --data-root or set up Kaggle inputs.")

    # Add ReferDINO / MapAnything repos to sys.path
    add_repo_paths(
        referdino_repo=cfg.referdino_repo,
        mapanything_repo=cfg.mapanything_repo,
    )

    # ---------- Dataset manifest ----------
    records = load_manifest(
        data_root=cfg.data_root,
        max_videos=cfg.max_videos,
        default_prompt=cfg.default_prompt,
        use_prompt_csv=cfg.use_prompt_csv_if_found,
    )

    # ---------- Preprocessing modules ----------
    keye        = KeyeModule(cfg.keye_model,        cfg.device, cfg.run_keye,        cfg.strict_external_models)
    referdino   = ReferDinoModule(cfg.referdino_repo, cfg.referdino_ckpt, cfg.device, cfg.run_referdino,   cfg.strict_external_models)
    mapanything = MapAnythingModule(cfg.mapanything_repo, cfg.mapanything_model,      cfg.device, cfg.run_mapanything, cfg.strict_external_models)

    # ---------- Preprocess ----------
    if getattr(args, "preprocess", True):
        processed_files = run_preprocess(
            records=records,
            cfg=cfg,
            keye=keye,
            referdino=referdino,
            mapanything=mapanything,
        )
    else:
        processed_files = sorted(cfg.proc_dir.glob("*.pt"))
        print(f"Skipping preprocess. Found {len(processed_files)} cached samples.")

    if getattr(args, "preprocess_only", False):
        print("preprocess-only mode finished.")
        return {"cfg": cfg, "processed_files": processed_files}

    # ---------- DataLoaders ----------
    all_files = sorted(cfg.proc_dir.glob("*.pt"))
    if len(all_files) < cfg.train_videos:
        raise RuntimeError(
            f"Need at least {cfg.train_videos} processed training samples, "
            f"found {len(all_files)} in {cfg.proc_dir}"
        )

    train_files, val_files, val_is_proxy = split_files(all_files, cfg.train_videos, cfg.test_videos)
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        train_files=train_files,
        val_files=val_files,
        batch_size=cfg.batch_size,
        num_workers=cfg.effective_num_workers,
        default_prompt=cfg.default_prompt,
        pin_memory=(cfg.device.type == "cuda"),
    )

    # Sanity check on first batch
    b0 = next(iter(train_loader))
    for meta in b0["module_meta"]:
        for mod_name in ["referdino", "mapanything"]:
            src = str(meta.get(mod_name, {}).get("source", ""))
            if any(x in src for x in ["fallback", "optical_flow", "disabled", "failed"]):
                raise RuntimeError(
                    f"Strict module check failed: {mod_name} source={src}"
                )
    print("Strict preprocessed-module check OK for first train batch.")

    # ---------- Model ----------
    model = WanSpatiaTrainer(cfg).to(cfg.device)

    # Initialise control branch
    with torch.no_grad():
        model.initialize_control_from_batch(b0)

    if not cfg.use_wan2_backbone or cfg.allow_toy_adapter:
        raise RuntimeError("Config invalid: strict Wan2.2 mode required.")

    # ---------- Training ----------
    stage = getattr(args, "stage", "both")

    if stage in {"stage1", "both"}:
        print("\n===== TRAIN STAGE 1 =====")
        set_stage1_trainable(model)
        train_loop(
            "stage1_wan_spatia_control", model,
            train_loader, val_loader,
            cfg.max_train_steps_stage1, cfg.lr_stage1,
            cfg, prompt_vocab=train_ds.prompt_to_id,
        )

    if stage in {"stage2", "both"}:
        print("\n===== TRAIN STAGE 2 =====")
        set_stage2_trainable(model)
        train_loop(
            "stage2_wan_lora", model,
            train_loader, val_loader,
            cfg.max_train_steps_stage2, cfg.lr_stage2,
            cfg, prompt_vocab=train_ds.prompt_to_id,
        )

    # ---------- Optional extras ----------
    benchmark_summary: dict = {}

    if getattr(args, "run_benchmark", False):
        print("\n===== BENCHMARK =====")
        benchmark_max = min(20, max(1, len(val_loader)))
        benchmark_df, benchmark_summary = run_benchmark(model, val_loader, cfg, max_batches=benchmark_max)
        print("Benchmark summary:")
        print(json.dumps(benchmark_summary, indent=2))
        save_benchmark_results(benchmark_df, benchmark_summary, cfg.ckpt_dir)

    if getattr(args, "save_sample", False):
        save_proxy_sample(
            model, val_loader,
            cfg.sample_dir / "wan_spatia_reconstruction.mp4",
            cfg, t_value=0.2,
        )

    # ---------- Final save ----------
    _final_save(cfg, model, train_ds, benchmark_summary)

    if getattr(args, "audit", False):
        _paper_audit(cfg, model)

    print("\nTraining pipeline finished.")
    print("Final checkpoint dir:", cfg.ckpt_dir)

    return {
        "cfg":               cfg,
        "model":             model,
        "train_ds":          train_ds,
        "val_ds":            val_ds,
        "benchmark_summary": benchmark_summary,
    }
