import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation (LoRA).
    Used in Stage 2 to fine-tune main blocks efficiently.

    W_out = W_frozen + (B @ A) * scale
    where A ∈ R^{rank×in}, B ∈ R^{out×rank}

    Base weight (W) is frozen; only lora_A and lora_B are trained.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 64, alpha: float = 64.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)   # LoRA delta = 0 at init ✓

        # Freeze base weight — only adapters are trained
        self.linear.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(x))
