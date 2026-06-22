"""
spatia_pipeline/data/dataset.py
---------------------------------
PyTorch Dataset + DataLoader factory for preprocessed Spatia samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SpatiaFullDataset(Dataset):
    """
    Dataset over preprocessed ``.pt`` files saved by the preprocess stage.

    Each file should contain a dict with at minimum:
        prev, target, control, memory, dynamic_mask, reference (tensors)
        prompt (str), id (str)
    """

    def __init__(self, files: List[Path | str], default_prompt: str = "") -> None:
        self.files = [Path(f) for f in files]
        self.default_prompt = default_prompt

        # Build prompt vocabulary from saved files
        prompts: List[str] = []
        for f in self.files:
            item = torch.load(f, map_location="cpu", weights_only=False)
            prompts.append(item.get("prompt", self.default_prompt))

        unique = sorted(set(prompts))
        self.prompt_to_id: Dict[str, int] = {p: i for i, p in enumerate(unique)}
        self.id_to_prompt: Dict[int, str] = {i: p for p, i in self.prompt_to_id.items()}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        item   = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        prompt = item.get("prompt", self.default_prompt)
        return {
            "prev":         item["prev"].float(),
            "target":       item["target"].float(),
            "control":      item["control"].float(),
            "memory":       item["memory"].float(),
            "dynamic_mask": item["dynamic_mask"].float(),
            "reference":    item["reference"].float(),
            "prompt_id":    torch.tensor(
                self.prompt_to_id.get(prompt, 0), dtype=torch.long
            ),
            "prompt_text":  prompt,
            "id":           item.get("id", self.files[idx].stem),
            "module_meta":  item.get("module_meta", {}),
        }


def collate_fn(batch: List[dict]) -> dict:
    tensor_keys = ["prev", "target", "control", "memory", "dynamic_mask", "reference", "prompt_id"]
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    out["prompt_text"] = [b["prompt_text"] for b in batch]
    out["id"]          = [b["id"] for b in batch]
    out["module_meta"] = [b["module_meta"] for b in batch]
    return out


def build_dataloaders(
    train_files: List[Path],
    val_files: List[Path],
    batch_size: int,
    num_workers: int,
    default_prompt: str,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, SpatiaFullDataset, SpatiaFullDataset]:
    """
    Create train + val DataLoaders along with their underlying datasets.

    Returns:
        train_loader, val_loader, train_ds, val_ds
    """
    train_ds = SpatiaFullDataset(train_files, default_prompt=default_prompt)
    val_ds   = SpatiaFullDataset(val_files,   default_prompt=default_prompt)

    # Share the prompt vocabulary so val ids are consistent
    val_ds.prompt_to_id = train_ds.prompt_to_id
    val_ds.id_to_prompt = train_ds.id_to_prompt

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent,
    )

    print(
        f"Train: {len(train_ds)}  Val: {len(val_ds)}"
        f"  Prompts: {len(train_ds.prompt_to_id)}"
    )
    return train_loader, val_loader, train_ds, val_ds


def split_files(
    all_files: List[Path],
    train_videos: int,
    test_videos: int,
) -> Tuple[List[Path], List[Path], bool]:
    """
    Split a flat list of ``.pt`` files into train / val.

    When fewer than ``train_videos + test_videos`` files exist, falls back to
    a small train-proxy validation set (to avoid crashing on 100-video runs).

    Returns:
        train_files, val_files, val_is_train_proxy
    """
    train_files = all_files[:train_videos]
    holdout     = all_files[train_videos: train_videos + test_videos]

    if holdout:
        return train_files, holdout, False

    proxy_n = min(5, len(train_files))
    print(
        f"WARN: no extra validation files found; "
        f"using {proxy_n} training samples as validation proxy."
    )
    return train_files, train_files[:proxy_n], True
