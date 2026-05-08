import os
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, optimizer, stage: int,
                    loss: float, save_dir: str, name: str) -> str:
    """Save a training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pt")
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": stage,
        "loss":  loss,
    }, path)
    print(f"  Checkpoint saved → {path}")
    return path


def load_checkpoint(model: nn.Module, optimizer, path: str, device: torch.device):
    """Load a training checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  Checkpoint loaded ← {path}  (stage={ckpt['stage']}, loss={ckpt['loss']:.6f})")
    return ckpt
