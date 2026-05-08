import torch
import torch.nn.functional as F
from configs.config import SpatiaConfig


def logit_normal_sample(shape, device: torch.device,
                         mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """
    Sample timestep t from logit-normal distribution (as used in Spatia / Flow Matching).
        u ~ N(mu, sigma)
        t = sigmoid(u)  →  t ∈ (0, 1)
    """
    u = torch.randn(shape, device=device) * sigma + mu
    return torch.sigmoid(u)


def flow_matching_loss(model, batch: dict,
                       cfg: SpatiaConfig, device: torch.device) -> torch.Tensor:
    """
    Compute Flow Matching MSE loss.

    Forward process (linear interpolation):
        x_0 ~ N(0, I)
        x_t = (1-t) * x_0 + t * x_T

    Ground-truth velocity:
        u_t = dx_t / dt = x_T - x_0

    Training objective:
        L = E[ || v_θ(x_t, cond, t) - u_t ||² ]

    Preceding-frame augmentation (paper Section 6):
        Add low-noise perturbation to x_P during training to bridge
        the train/inference distribution gap:
            t_aug ~ Uniform[0, aug_t_max]
            x_P_aug = (1-t_aug) * x_P + t_aug * ε,  ε ~ N(0, I)
    """
    x_T   = batch["x_T"].to(device)
    x_P   = batch["x_P"].to(device)
    x_R   = batch["x_R"].to(device)
    x_S_T = batch["x_S_T"].to(device)
    x_S_P = batch["x_S_P"].to(device)
    text  = batch["text"].to(device)
    B     = x_T.shape[0]

    # ── Preceding-frame augmentation ──────────────────────────────────
    t_aug        = torch.rand(B, device=device) * cfg.aug_t_max   # [B]
    noise_P      = torch.randn_like(x_P)
    x_P_aug      = ((1 - t_aug[:, None, None]) * x_P
                    + t_aug[:, None, None] * noise_P)

    # ── Sample timestep from logit-normal ─────────────────────────────
    t = logit_normal_sample((B,), device=device)   # [B]

    # ── Linear interpolation ──────────────────────────────────────────
    x_0  = torch.randn_like(x_T)
    x_t  = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_T

    # ── Ground-truth velocity ─────────────────────────────────────────
    u_t  = x_T - x_0

    # ── Predicted velocity ────────────────────────────────────────────
    v_t  = model(x_t, x_P_aug, x_R, x_S_T, x_S_P, text, t)

    return F.mse_loss(v_t, u_t)
