"""
spatia_pipeline/training/trainer.py
--------------------------------------
Training utilities: stage setup, evaluate, train_loop.
"""

from __future__ import annotations

import math
import time
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spatia_pipeline.config import SpatiaConfig
from spatia_pipeline.training.checkpoint import (
    keep_latest_checkpoints,
    safe_torch_save,
)


# ---------------------------------------------------------------------------
# Stage-mode helpers
# ---------------------------------------------------------------------------

def apply_stage_module_modes(model) -> None:
    """
    Keep frozen Wan modules in eval mode; only the intended trainable
    branch is set to train mode.

    Called after every gradient step and before/after validation to prevent
    a failed validation from leaving model.stage='eval'.
    """
    if getattr(model, "vae", None) is not None:
        model.vae.eval()
    if getattr(model, "text_encoder", None) is not None:
        model.text_encoder.eval()
    if getattr(model, "control", None) is not None:
        model.control.train(model.stage == "stage1")
    if getattr(model, "transformer", None) is not None:
        model.transformer.train(model.stage == "stage2")


def set_stage1_trainable(model) -> None:
    """Freeze everything except the latent control branch."""
    model.stage = "stage1"
    for p in model.parameters():
        p.requires_grad = False
    for p in model.control.parameters():
        p.requires_grad = True
    apply_stage_module_modes(model)


def set_stage2_trainable(model) -> None:
    """Freeze control branch; enable gradients for Wan transformer LoRA params only."""
    model.stage = "stage2"
    for p in model.parameters():
        p.requires_grad = False
    for p in model.control.parameters():
        p.requires_grad = False

    lora_count = 0
    for n, p in model.transformer.named_parameters():
        if "lora" in n.lower():
            p.requires_grad = True
            lora_count += p.numel()

    if lora_count == 0:
        raise RuntimeError("No LoRA parameters found in Wan transformer for Stage 2.")

    apply_stage_module_modes(model)
    print("Stage2 LoRA trainable params:", lora_count)


# ---------------------------------------------------------------------------
# GradScaler factory
# ---------------------------------------------------------------------------

def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, loader: DataLoader, cfg: SpatiaConfig) -> dict:
    """
    Run a short validation pass.

    Restores ``model.stage`` and train/eval mode in a ``finally`` block so
    a failed pass can never leave the model stuck in eval stage (which would
    silently disable LoRA gradients on the next training step).
    """
    old_stage    = getattr(model, "stage", None)
    old_training = model.training
    losses, psnrs = [], []

    try:
        model.stage = "eval"
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= cfg.eval_max_batches:
                    break

                loss = model.training_loss(batch)
                if torch.isfinite(loss):
                    losses.append(float(loss.detach().cpu()))

                pred, target, _ = model.reconstruct_video(batch, t_value=0.2)
                mse  = F.mse_loss((pred + 1) / 2, (target + 1) / 2).item()
                psnr = -10 * math.log10(max(mse, 1e-8))
                psnrs.append(psnr)

        val_loss  = float(np.mean(losses)) if losses else float("nan")
        psnr_val  = float(np.mean(psnrs))  if psnrs  else float("nan")
        return {"val_loss": val_loss, "psnr_proxy": psnr_val}

    finally:
        if old_stage is not None:
            model.stage = old_stage
        model.train(old_training)
        apply_stage_module_modes(model)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(
    stage_name: str,
    model,
    loader: DataLoader,
    val_loader: DataLoader,
    max_steps: int,
    lr: float,
    cfg: SpatiaConfig,
    prompt_vocab: Optional[dict] = None,
) -> None:
    """
    Generic training loop used for both Stage 1 and Stage 2.

    Args:
        stage_name:   human-readable name logged in output (e.g. "stage1_wan_spatia_control")
        model:        WanSpatiaTrainer instance (already staged)
        loader:       training DataLoader
        val_loader:   validation DataLoader
        max_steps:    total gradient-update steps
        lr:           base learning rate (scaled by scheduler)
        cfg:          pipeline configuration
        prompt_vocab: prompt_to_id dict for checkpoint (optional)
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)

    if n_trainable == 0:
        raise RuntimeError(f"{stage_name}: no trainable parameters")

    print(f"{stage_name} trainable params: {n_trainable:,}")

    # Keep stage explicit — prevents a failed validation from disabling LoRA grads
    stage_name_l = stage_name.lower()
    train_stage  = (
        "stage2" if "stage2" in stage_name_l
        else ("stage1" if "stage1" in stage_name_l else getattr(model, "stage", "stage1"))
    )
    model.stage = train_stage
    apply_stage_module_modes(model)

    opt = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=cfg.optim_weight_decay,
        eps=cfg.optim_eps,
        foreach=False,
    )
    scaler           = _make_grad_scaler(cfg.use_grad_scaler)
    grad_clip_val    = cfg.grad_clip(stage_name)
    iterator: Iterator = iter(loader)
    running          = []
    start            = time.time()
    consecutive_bad  = 0
    skipped_steps    = 0
    lr_safety_factor = 1.0

    if cfg.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, max_steps + 1):
        # Re-assert every step (cheap + robust)
        model.stage = train_stage
        apply_stage_module_modes(model)

        current_lr = lr * cfg.lr_scale(step, max_steps) * lr_safety_factor
        for group in opt.param_groups:
            group["lr"] = current_lr

        opt.zero_grad(set_to_none=True)
        total    = 0.0
        bad_step = False

        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch    = next(iterator)

            with torch.autocast(
                device_type=cfg.device.type,
                dtype=cfg.amp_dtype,
                enabled=(cfg.device.type == "cuda"),
            ):
                loss = model.training_loss(batch) / cfg.grad_accum_steps

            if not torch.isfinite(loss):
                val = float(loss.detach().cpu()) if loss.numel() == 1 else loss
                print(f"[{stage_name}] step {step}: non-finite loss, skip: {val}")
                bad_step = True
                break

            if not loss.requires_grad:
                n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
                lora_t = sum(
                    p.numel()
                    for n, p in model.transformer.named_parameters()
                    if "lora" in n.lower() and p.requires_grad
                )
                raise RuntimeError(
                    f"{stage_name}: loss has no grad_fn. "
                    f"model.stage={getattr(model, 'stage', None)!r}, "
                    f"trainable_now={n_t}, lora_trainable_now={lora_t}. "
                    "Validation likely left model in eval/non-train stage."
                )

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total += float(loss.detach().cpu()) * cfg.grad_accum_steps

        # --- Bad step handling ---
        if bad_step:
            skipped_steps   += 1
            consecutive_bad += 1
            opt.zero_grad(set_to_none=True)
            if cfg.device.type == "cuda":
                torch.cuda.empty_cache()
            if consecutive_bad % 3 == 0:
                lr_safety_factor = max(lr_safety_factor * 0.5, 0.05)
                print(f"[{stage_name}] LR backoff: safety_factor={lr_safety_factor:.3f}")
            if consecutive_bad >= cfg.max_consecutive_bad_steps:
                print(f"[{stage_name}] WARN: {consecutive_bad} consecutive bad steps; resetting counter.")
                consecutive_bad = 0
            continue

        # --- Gradient clipping ---
        if scaler is not None:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_val)

        if not torch.isfinite(grad_norm):
            skipped_steps   += 1
            consecutive_bad += 1
            print(f"[{stage_name}] step {step}: non-finite grad norm, skip")
            opt.zero_grad(set_to_none=True)
            if cfg.device.type == "cuda":
                torch.cuda.empty_cache()
            if consecutive_bad % 3 == 0:
                lr_safety_factor = max(lr_safety_factor * 0.5, 0.05)
                print(f"[{stage_name}] LR backoff: safety_factor={lr_safety_factor:.3f}")
            continue

        # --- Optimizer step ---
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        opt.zero_grad(set_to_none=True)
        consecutive_bad = 0
        running.append(total)

        # --- Logging ---
        if step % cfg.log_every == 0 or step == 1:
            loss_msg = float(np.mean(running[-cfg.log_every:])) if running else float("nan")
            msg = (
                f"[{stage_name}] step {step}/{max_steps}"
                f" loss={loss_msg:.4f}"
                f" grad_norm={float(grad_norm):.4f}"
                f" lr={current_lr:.2e}"
                f" skipped={skipped_steps}"
            )
            if cfg.device.type == "cuda":
                msg += f" peak_vram={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
            print(msg)

        # --- Validation ---
        if step % cfg.val_every == 0 or step == max_steps:
            try:
                print("VAL", evaluate(model, val_loader, cfg))
            except Exception as e:
                print(f"WARN: validation skipped; training state restored: {type(e).__name__}: {e}")
                if cfg.device.type == "cuda":
                    torch.cuda.empty_cache()
            finally:
                model.stage = train_stage
                apply_stage_module_modes(model)

        # --- Checkpoint ---
        if step % cfg.save_every == 0 or step == max_steps:
            pattern = f"{stage_name}_step_*_wan_trainable.pt"

            keep_latest_checkpoints(
                cfg.ckpt_dir, pattern=pattern, keep=max(cfg.keep_last_ckpts - 1, 1)
            )

            ckpt = cfg.ckpt_dir / f"{stage_name}_step_{step}_wan_trainable.pt"
            payload = {
                "stage":           stage_name,
                "step":            step,
                "trainable_state": model.export_trainable_state(),
                "prompt_to_id":    prompt_vocab or {},
                "note": (
                    "Wan2.2-backed Spatia-style latent control + Wan transformer LoRA. "
                    "Stores trainable delta only, not full Wan2.2 weights."
                ),
                "skipped_steps":    skipped_steps,
                "amp_dtype":        str(cfg.amp_dtype),
                "lr_safety_factor": lr_safety_factor,
            }
            safe_torch_save(payload, ckpt, min_free_gb=cfg.min_free_gb_for_save)

            keep_latest_checkpoints(cfg.ckpt_dir, pattern=pattern, keep=cfg.keep_last_ckpts)

            if cfg.device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = round((time.time() - start) / 60, 2)
    print(f"{stage_name} done in {elapsed} min | skipped_steps: {skipped_steps}")
