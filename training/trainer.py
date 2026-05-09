import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
try:
    from torch.optim.lr_scheduler import LRScheduler   # PyTorch >= 2.4 (public)
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # PyTorch < 2.4
from configs.config import SpatiaConfig
from .loss import flow_matching_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LRScheduler,
    cfg: SpatiaConfig,
    device: torch.device,
    stage: int,
    max_iters: int | None = None,
) -> float:
    """
    Run one pass over the dataloader (or up to max_iters steps).

    Args:
        model       : Spatia model
        dataloader  : training DataLoader
        optimizer   : AdamW optimizer (already configured for current stage)
        scheduler   : LR scheduler — stepped once per optimizer step
        cfg         : SpatiaConfig
        device      : torch device
        stage       : 1 (ControlNet) or 2 (LoRA)
        max_iters   : stop early after this many steps (useful when
                      stage1_iters < len(dataloader))

    Returns:
        Average loss over all steps taken.
    """
    model.train()
    total_loss = 0.0
    num_steps  = 0

    for step, batch in enumerate(dataloader):
        if max_iters is not None and step >= max_iters:
            break

        optimizer.zero_grad()
        loss = flow_matching_loss(model, batch, cfg, device)
        loss.backward()

        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg.grad_clip,
        )
        optimizer.step()
        scheduler.step()   # per-step: CosineAnnealingLR expects this

        total_loss += loss.item()
        num_steps  += 1

        if step % cfg.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  [Stage {stage}] step {step:5d} | loss = {loss.item():.6f} | lr = {lr:.2e}")

    avg_loss = total_loss / max(num_steps, 1)
    print(f"  [Stage {stage}] epoch avg loss = {avg_loss:.6f}  ({num_steps} steps)")
    return avg_loss
