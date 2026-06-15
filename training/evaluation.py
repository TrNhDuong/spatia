import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def latent_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def clone_batch_for_eval(
    batch: dict[str, torch.Tensor],
    cfg,
    *,
    use_scene: bool = True,
    use_reference: bool = True,
    ref_frames: int | None = None,
) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in batch.items()}

    if not use_scene:
        out["x_S_T"].zero_()
        out["x_S_P"].zero_()

    if not use_reference:
        out["x_R"].zero_()
    elif ref_frames is not None:
        h_lat = cfg.height // cfg.spatial_downsample
        w_lat = cfg.width // cfg.spatial_downsample
        tokens_per_ref = h_lat * w_lat
        keep = max(0, min(ref_frames, cfg.max_ref_frames)) * tokens_per_ref
        out["x_R"][:, keep:].zero_()

    return out


@torch.no_grad()
def generate_latents(
    model,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    ode_steps: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    model.eval()
    target = batch["x_T"]
    bsz = target.shape[0]
    x_pred = torch.randn_like(target)
    dt = 1.0 / ode_steps

    for step in range(ode_steps):
        t = torch.full((bsz,), step * dt, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            velocity = model(
                x_pred,
                batch["x_P"],
                batch["x_R"],
                batch["x_S_T"],
                batch["x_S_P"],
                batch["text"],
                t,
            )
        x_pred = x_pred + velocity.float() * dt

    return x_pred


def _tokens_to_vae_latent(tokens: torch.Tensor, cfg) -> torch.Tensor:
    bsz, n_tokens, channels = tokens.shape
    h_lat = cfg.height // cfg.spatial_downsample
    w_lat = cfg.width // cfg.spatial_downsample
    frame_tokens = h_lat * w_lat
    if n_tokens % frame_tokens != 0:
        raise ValueError(
            f"Token count {n_tokens} is not divisible by latent grid {h_lat}x{w_lat}."
        )
    t_lat = n_tokens // frame_tokens
    return (
        tokens.reshape(bsz, t_lat, h_lat, w_lat, channels)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


@torch.no_grad()
def decode_latent_tokens(vae, tokens: torch.Tensor, cfg) -> torch.Tensor:
    if vae is None or getattr(vae, "model", None) is None:
        raise RuntimeError("Wan VAE is required for paper-style pixel metrics.")

    model = vae.model
    vae_device = next(model.parameters()).device
    vae_dtype = next(model.parameters()).dtype
    latents = _tokens_to_vae_latent(tokens, cfg).to(device=vae_device, dtype=vae_dtype)

    decoded = model.decode(latents)
    frames = decoded.sample if hasattr(decoded, "sample") else decoded[0]
    frames = frames.detach().float().clamp(-1, 1)

    if frames.ndim != 5:
        raise ValueError(f"Expected decoded video tensor with 5 dims, got {frames.shape}.")
    if frames.shape[1] == 3:
        frames = frames.permute(0, 2, 1, 3, 4)
    elif frames.shape[2] != 3:
        raise ValueError(f"Could not identify RGB channel in decoded tensor {frames.shape}.")

    return ((frames + 1.0) / 2.0).cpu().clamp(0, 1)


def _as_numpy_image(frame: torch.Tensor) -> np.ndarray:
    return (
        frame.detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        .clip(0.0, 1.0)
    )


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for paper-style SSIM. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return float(
        structural_similarity(
            _as_numpy_image(target),
            _as_numpy_image(pred),
            data_range=1.0,
            channel_axis=-1,
        )
    )


def build_lpips_model(device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "lpips is required for paper-style LPIPS. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


@torch.no_grad()
def _lpips_video(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model,
    device: torch.device,
    resize_to: int | None,
) -> float:
    if lpips_model is None:
        return float("nan")

    scores: list[float] = []
    bsz, frames = pred.shape[:2]
    for b in range(bsz):
        for t in range(frames):
            p = pred[b, t].unsqueeze(0).to(device) * 2.0 - 1.0
            y = target[b, t].unsqueeze(0).to(device) * 2.0 - 1.0
            if resize_to is not None:
                p = F.interpolate(p, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
                y = F.interpolate(y, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
            scores.append(float(lpips_model(p, y).mean().detach().cpu()))
    return float(np.mean(scores)) if scores else float("nan")


def video_quality_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lpips_model=None,
    lpips_device: torch.device | None = None,
    lpips_resize_to: int | None = None,
) -> dict[str, float]:
    pred = pred[:, : target.shape[1]]
    target = target[:, : pred.shape[1]]

    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    for b in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            psnr_scores.append(_psnr(pred[b, t], target[b, t]))
            ssim_scores.append(_ssim(pred[b, t], target[b, t]))

    lpips_score = _lpips_video(
        pred,
        target,
        lpips_model,
        lpips_device or torch.device("cpu"),
        lpips_resize_to,
    )
    return {
        "psnr": float(np.mean(psnr_scores)) if psnr_scores else float("nan"),
        "ssim": float(np.mean(ssim_scores)) if ssim_scores else float("nan"),
        "lpips": lpips_score,
    }


def _orb_match_count(a: torch.Tensor, b: torch.Tensor) -> int:
    import cv2

    def gray_u8(x: torch.Tensor) -> np.ndarray:
        arr = (_as_numpy_image(x) * 255.0).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray_u8(a), None)
    kp2, des2 = orb.detectAndCompute(gray_u8(b), None)
    if des1 is None or des2 is None or not kp1 or not kp2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    good = [m for m in matches if m.distance <= 64]
    return len(good)


def match_accuracy_proxy(initial: torch.Tensor, final: torch.Tensor) -> float:
    scores: list[float] = []
    for b in range(initial.shape[0]):
        denom = max(_orb_match_count(initial[b], initial[b]), 1)
        score = _orb_match_count(initial[b], final[b]) / denom
        scores.append(float(max(0.0, min(score, 1.0))))
    return float(np.mean(scores)) if scores else float("nan")


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.nanmean([row[k] for row in rows])) for k in keys}


@torch.no_grad()
def evaluate_paper_metrics(
    model,
    loader,
    cfg,
    *,
    device: torch.device,
    vae=None,
    ode_steps: int = 20,
    max_samples: int = 100,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    use_scene: bool = True,
    use_reference: bool = True,
    ref_frames: int | None = None,
    lpips_model=None,
    lpips_resize_to: int | None = None,
) -> dict[str, float]:
    latent_rows: list[dict[str, float]] = []
    realestate_rows: list[dict[str, float]] = []
    closed_loop_rows: list[dict[str, float]] = []
    seen = 0

    for batch in loader:
        if seen >= max_samples:
            break
        batch = move_batch(batch, device)
        batch = clone_batch_for_eval(
            batch,
            cfg,
            use_scene=use_scene,
            use_reference=use_reference,
            ref_frames=ref_frames,
        )
        pred_latents = generate_latents(
            model,
            batch,
            device=device,
            ode_steps=ode_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        target_latents = batch["x_T"]
        latent_rows.append({"latent_psnr": latent_psnr(pred_latents, target_latents)})

        if vae is not None:
            pred_video = decode_latent_tokens(vae, pred_latents, cfg)
            target_video = decode_latent_tokens(vae, target_latents, cfg)
            realestate_rows.append(
                video_quality_metrics(
                    pred_video,
                    target_video,
                    lpips_model=lpips_model,
                    lpips_device=device,
                    lpips_resize_to=lpips_resize_to,
                )
            )

            init_video = decode_latent_tokens(vae, batch["x_P"], cfg)
            initial_frame = init_video[:, 0]
            final_frame = pred_video[:, -1]
            closed_metrics = video_quality_metrics(
                final_frame.unsqueeze(1),
                initial_frame.unsqueeze(1),
                lpips_model=lpips_model,
                lpips_device=device,
                lpips_resize_to=lpips_resize_to,
            )
            closed_loop_rows.append(
                {
                    "psnr_c": closed_metrics["psnr"],
                    "ssim_c": closed_metrics["ssim"],
                    "lpips_c": closed_metrics["lpips"],
                    "match_acc": match_accuracy_proxy(initial_frame, final_frame),
                }
            )

        seen += target_latents.shape[0]

    result: dict[str, float] = {"samples": float(min(seen, max_samples))}
    result.update(_mean_dict(latent_rows))
    result.update({f"realestate_{k}": v for k, v in _mean_dict(realestate_rows).items()})
    result.update(_mean_dict(closed_loop_rows))
    return result


def write_metrics_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
