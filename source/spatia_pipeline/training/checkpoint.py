"""
spatia_pipeline/training/checkpoint.py
----------------------------------------
Atomic checkpoint save / rotation utilities.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Union

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_gb(path: Path) -> float:
    _, _, free = shutil.disk_usage(str(path))
    return free / 1024 ** 3


def _to_cpu_detached(obj):
    """Recursively move tensors to CPU before checkpoint serialisation."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_detached(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_detached(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_detached(v) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cleanup_tmp_checkpoints(ckpt_dir: Path) -> None:
    """Remove any leftover ``.pt.tmp`` files from a previously interrupted save."""
    for p in Path(ckpt_dir).glob("*.pt.tmp"):
        try:
            print("Remove unfinished temp checkpoint:", p.name)
            p.unlink()
        except Exception as e:
            print(f"WARN cannot remove temp checkpoint {p}: {type(e).__name__}: {e}")


def keep_latest_checkpoints(
    ckpt_dir: Path,
    pattern: str = "*.pt",
    keep: int = 2,
) -> None:
    """
    Delete old checkpoints in ``ckpt_dir`` matching ``pattern``,
    keeping only the ``keep`` most-recently-modified files.
    """
    ckpt_dir = Path(ckpt_dir)
    keep = max(int(keep), 0)
    files = sorted(
        ckpt_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in files[keep:]:
        try:
            print("Remove old checkpoint:", p.name)
            p.unlink()
        except Exception as e:
            print(f"WARN cannot remove checkpoint {p}: {type(e).__name__}: {e}")


def safe_torch_save(
    obj,
    path: Union[str, Path],
    min_free_gb: float = 1.0,
) -> None:
    """
    Atomically save ``obj`` to ``path``:
      1. Move all tensors to CPU.
      2. Write to a ``.tmp`` file.
      3. ``os.replace`` (atomic rename) to the final path.
      4. Print disk-usage before/after.

    Raises ``RuntimeError`` if less than ``min_free_gb`` is available.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_tmp_checkpoints(path.parent)

    free_before = _free_gb(path.parent)
    print(f"Free disk before save: {free_before:.2f} GB")
    if free_before < min_free_gb:
        raise RuntimeError(
            f"Not enough disk space. Free: {free_before:.2f} GB, "
            f"required: {min_free_gb:.2f} GB."
        )

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        torch.save(_to_cpu_detached(obj), tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    print("Saved", path)
    print(f"Free disk after save: {_free_gb(path.parent):.2f} GB")
