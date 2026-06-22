"""
spatia_pipeline/model/wan_trainer.py
--------------------------------------
WanSpatiaTrainer — the main training model.

Wraps a Wan2.2 DiffusionPipeline and adds:
  - A LatentSpatiaControlNet control branch (Stage 1)
  - PEFT LoRA adapters on the Wan transformer (Stage 2)
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatia_pipeline.config import SpatiaConfig
from spatia_pipeline.model.control_net import LatentSpatiaControlNet


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def first_parameter_device_dtype(
    module: nn.Module,
) -> Tuple[torch.device, torch.dtype]:
    p = next(module.parameters())
    return p.device, p.dtype


def count_module_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def extract_tensor_from_output(out) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            if torch.is_tensor(x):
                return x
    for attr in ["sample", "pred", "prediction", "hidden_states", "last_hidden_state"]:
        if hasattr(out, attr):
            x = getattr(out, attr)
            if torch.is_tensor(x):
                return x
    raise TypeError(f"Cannot extract tensor from output type {type(out)}")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# WanSpatiaTrainer
# ---------------------------------------------------------------------------

class WanSpatiaTrainer(nn.Module):
    """
    Wan2.2-backed Spatia training model.

    Stages:
        stage1 — train only ``control`` (LatentSpatiaControlNet)
        stage2 — freeze control, fine-tune Wan transformer with PEFT LoRA
    """

    def __init__(self, cfg: SpatiaConfig) -> None:
        super().__init__()

        if cfg.wan_dir is None:
            raise ValueError("cfg.wan_dir must be set before creating WanSpatiaTrainer")

        self.cfg     = cfg
        self.wan_dir = Path(cfg.wan_dir)

        # --- Load Wan2.2 pipeline ---
        try:
            from diffusers import DiffusionPipeline  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "diffusers is required for Wan2.2 backbone training."
            ) from e

        self.pipe = DiffusionPipeline.from_pretrained(
            str(self.wan_dir),
            torch_dtype=cfg.backbone_dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.pipe.to(cfg.device)

        self.transformer   = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
        self.vae           = getattr(self.pipe, "vae", None)
        self.text_encoder  = getattr(self.pipe, "text_encoder", None)
        self.tokenizer     = getattr(self.pipe, "tokenizer", None)
        self.scheduler     = getattr(self.pipe, "scheduler", None)

        missing = [
            n for n in ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
            if getattr(self, n) is None
        ]
        if missing:
            raise RuntimeError(
                f"Wan pipeline missing required components: {missing}. "
                f"Components: {list(self.pipe.components.keys())}"
            )

        # Gradient checkpointing
        if cfg.enable_gradient_checkpointing:
            if hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()
                print("Transformer gradient checkpointing: enabled")
            elif hasattr(self.transformer, "gradient_checkpointing"):
                self.transformer.gradient_checkpointing = True
                print("Transformer gradient checkpointing flag: enabled")

        # Freeze entire Wan backbone
        for module in [self.transformer, self.vae, self.text_encoder]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

        # VAE memory optimisations
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()

        # Runtime state
        self._vae_layout:     Optional[str] = None
        self.control:         Optional[LatentSpatiaControlNet] = None
        self.control_channels: Optional[int] = None
        self.stage:           str = "stage1"
        self.lora_target_modules: List[str] = []

        self._inject_lora()

        # Audit
        wan_params = (
            count_module_params(self.transformer)
            + count_module_params(self.vae)
            + count_module_params(self.text_encoder)
        )
        print("Wan train pipeline class:", type(self.pipe).__name__)
        print("Transformer class:", type(self.transformer).__name__)
        print("VAE class:", type(self.vae).__name__)
        print("Text encoder class:", type(self.text_encoder).__name__)
        print(f"Approx Wan component params: {wan_params / 1e9:.3f}B")

        if cfg.strict_wan_backbone and wan_params < 1_000_000_000:
            raise RuntimeError(
                f"Wan backbone check failed: only {wan_params / 1e6:.1f}M params detected."
            )

    # ------------------------------------------------------------------
    # LoRA injection
    # ------------------------------------------------------------------

    def _inject_lora(self) -> None:
        try:
            from peft import LoraConfig  # type: ignore
        except ImportError as e:
            if self.cfg.strict_wan_backbone:
                raise RuntimeError("peft is required for Stage 2 LoRA on Wan2.2 transformer.") from e
            print("WARN peft not available:", e)
            return

        linear_names = [n for n, m in self.transformer.named_modules() if isinstance(m, nn.Linear)]
        preferred_tails = [
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "q", "k", "v", "o", "proj", "proj_out", "fc1", "fc2",
        ]
        tails_present = {n.split(".")[-1] for n in linear_names}
        targets = [t for t in preferred_tails if t.split(".")[-1] in tails_present or t in linear_names]

        if not targets:
            attn_names = [n for n in linear_names if "attn" in n.lower() or "attention" in n.lower()]
            targets = sorted({n.split(".")[-1] for n in attn_names})[:8]

        if not targets:
            raise RuntimeError("Could not infer LoRA target modules inside Wan transformer.")

        self.lora_target_modules = sorted(set(targets))
        config = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            init_lora_weights=True,
            target_modules=self.lora_target_modules,
        )
        if hasattr(self.transformer, "add_adapter"):
            self.transformer.add_adapter(config)
        else:
            raise RuntimeError("Wan transformer does not support add_adapter; cannot attach PEFT LoRA.")

        for n, p in self.transformer.named_parameters():
            p.requires_grad = "lora" in n.lower()

        print("LoRA attached to Wan transformer target modules:", self.lora_target_modules)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _text_embeds(self, prompts: List[str]) -> torch.Tensor:
        device, dtype = first_parameter_device_dtype(self.text_encoder)
        tok = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.text_encoder(**tok)
        emb = extract_tensor_from_output(out)
        return emb.to(device=self.cfg.device, dtype=dtype)

    def _vae_encode_raw(self, video: torch.Tensor, layout: str) -> torch.Tensor:
        if layout == "BCTHW":
            x = video.permute(0, 2, 1, 3, 4).contiguous()
        elif layout == "BTCHW":
            x = video.contiguous()
        else:
            raise ValueError(layout)
        _, dtype = first_parameter_device_dtype(self.vae)
        x = x.to(device=self.cfg.device, dtype=dtype)
        out = self.vae.encode(x)
        z   = out.latent_dist.sample() if hasattr(out, "latent_dist") else extract_tensor_from_output(out)
        scale = float(getattr(getattr(self.vae, "config", object()), "scaling_factor", 1.0))
        return z * scale

    def _sanitize_latents(self, z: torch.Tensor, name: str = "latents") -> torch.Tensor:
        cfg = self.cfg
        z = z.float()
        if not torch.isfinite(z).all():
            print(f"WARN: non-finite values in {name}; applying nan_to_num.")
            z = torch.nan_to_num(z, nan=0.0, posinf=cfg.latent_clamp_value, neginf=-cfg.latent_clamp_value)
        return z.clamp(-cfg.latent_clamp_value, cfg.latent_clamp_value).to(device=cfg.device)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """video: [B, T, C, H, W] in [-1, 1] → latent [B, C', T', H', W']"""
        with torch.no_grad():
            if self._vae_layout is not None:
                return self._sanitize_latents(self._vae_encode_raw(video, self._vae_layout))
            errors = []
            for layout in ["BCTHW", "BTCHW"]:
                try:
                    z = self._vae_encode_raw(video, layout)
                    self._vae_layout = layout
                    z = self._sanitize_latents(z, "vae_latents")
                    print("VAE encode layout selected:", layout, "latent shape:", tuple(z.shape))
                    return z
                except Exception as e:
                    errors.append((layout, f"{type(e).__name__}: {e}"))
                    if self.cfg.device.type == "cuda":
                        torch.cuda.empty_cache()
            raise RuntimeError(f"Wan VAE encode failed for all layouts: {errors}")

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        scale = float(getattr(getattr(self.vae, "config", object()), "scaling_factor", 1.0))
        z     = latents / scale
        _, dtype = first_parameter_device_dtype(self.vae)
        with torch.no_grad():
            out = self.vae.decode(z.to(device=self.cfg.device, dtype=dtype))
        x = extract_tensor_from_output(out)
        if x.ndim == 5 and x.shape[1] in [1, 3, 4, 8, 16] and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.float().clamp(-1, 1)

    def make_condition_video(self, batch: dict) -> torch.Tensor:
        target = batch["target"]
        T = target.shape[1]
        prev     = batch["prev"]
        control  = batch["control"]
        memory   = batch["memory"]
        refs     = batch["reference"]
        prev_summary = prev.mean(dim=1, keepdim=True).repeat(1, T, 1, 1, 1)
        ref_summary  = refs.mean(dim=1, keepdim=True).repeat(1, T, 1, 1, 1)
        cond = 0.30 * prev_summary + 0.30 * control + 0.25 * memory + 0.15 * ref_summary
        return cond.clamp(-1, 1)

    def initialize_control_from_batch(self, batch: dict) -> Tuple:
        batch   = move_batch(batch, self.cfg.device)
        latents = self.encode_video(batch["target"])
        channels = latents.shape[1]
        self.control_channels = channels
        self.control = LatentSpatiaControlNet(
            channels=channels,
            width=self.cfg.control_width,
            depth=self.cfg.control_depth,
        ).to(self.cfg.device, dtype=torch.float32)
        print("Latent control initialized. latent shape:", tuple(latents.shape), "channels:", channels)
        print("Control branch params:", count_module_params(self.control))
        return latents.shape

    # ------------------------------------------------------------------
    # Transformer call
    # ------------------------------------------------------------------

    def _transformer_call(
        self,
        noisy_latents: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        grad_enabled: bool,
    ) -> torch.Tensor:
        tr  = self.transformer
        sig = inspect.signature(tr.forward)

        # Determine backbone dtype (use non-LoRA param dtype)
        tr_dtype = self.cfg.backbone_dtype
        try:
            for _n, _p in tr.named_parameters():
                if "lora" not in _n.lower():
                    tr_dtype = _p.dtype
                    break
        except Exception:
            pass

        noisy_latents  = noisy_latents.to(device=self.cfg.device, dtype=tr_dtype).contiguous()
        prompt_embeds  = prompt_embeds.to(device=self.cfg.device, dtype=tr_dtype)
        timestep       = (t * 1000).to(device=self.cfg.device, dtype=torch.float32)

        candidates: Dict = {
            "hidden_states":        noisy_latents,
            "sample":               noisy_latents,
            "latents":              noisy_latents,
            "x":                    noisy_latents,
            "timestep":             timestep,
            "timesteps":            timestep,
            "t":                    timestep,
            "encoder_hidden_states": prompt_embeds,
            "context":              prompt_embeds,
            "return_dict":          False,
        }
        kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}

        if not any(k in kwargs for k in ["hidden_states", "sample", "latents", "x"]):
            kwargs["hidden_states"] = noisy_latents
        if not any(k in kwargs for k in ["timestep", "timesteps", "t"]):
            kwargs["timestep"] = timestep
        if not any(k in kwargs for k in ["encoder_hidden_states", "context"]):
            kwargs["encoder_hidden_states"] = prompt_embeds

        with torch.set_grad_enabled(grad_enabled):
            out = tr(**kwargs)

        pred = extract_tensor_from_output(out)
        if pred.shape != noisy_latents.shape:
            if (pred.ndim == 5
                    and pred.shape[1] == noisy_latents.shape[2]
                    and pred.shape[2] == noisy_latents.shape[1]):
                pred = pred.permute(0, 2, 1, 3, 4).contiguous()
            if pred.shape != noisy_latents.shape:
                raise RuntimeError(
                    f"Wan transformer output shape {tuple(pred.shape)} "
                    f"!= latent shape {tuple(noisy_latents.shape)}"
                )
        return pred

    # ------------------------------------------------------------------
    # Loss + reconstruction
    # ------------------------------------------------------------------

    def training_loss(self, batch: dict) -> torch.Tensor:
        cfg    = self.cfg
        batch  = move_batch(batch, cfg.device)
        prompts = batch.get("prompt_text", [cfg.default_prompt] * batch["target"].shape[0])

        latents      = self.encode_video(batch["target"]).float().clamp(-cfg.latent_clamp_value, cfg.latent_clamp_value)
        cond_latents = self.encode_video(self.make_condition_video(batch)).float().clamp(-cfg.latent_clamp_value, cfg.latent_clamp_value)

        if self.control is None:
            self.initialize_control_from_batch(batch)

        B  = latents.shape[0]
        t  = torch.empty(B, device=cfg.device, dtype=torch.float32).uniform_(cfg.timestep_min, cfg.timestep_max)
        ex = (slice(None),) + (None,) * (latents.ndim - 1)

        noise        = torch.randn_like(latents).clamp(-cfg.noise_clamp_value, cfg.noise_clamp_value)
        noisy        = ((1 - t[ex]) * latents + t[ex] * noise).clamp(-cfg.latent_clamp_value, cfg.latent_clamp_value)
        target_v     = (noise - latents).float().clamp(-cfg.pred_clamp_value, cfg.pred_clamp_value)

        prompt_embeds = self._text_embeds(prompts)
        train_lora    = self.stage == "stage2"

        base_pred = self._transformer_call(noisy, t.float(), prompt_embeds, grad_enabled=train_lora)
        if self.stage == "stage1":
            base_pred = base_pred.detach()

        base_pred = torch.nan_to_num(base_pred.float(), nan=0.0,
                                     posinf=cfg.pred_clamp_value, neginf=-cfg.pred_clamp_value)
        base_pred = base_pred.clamp(-cfg.pred_clamp_value, cfg.pred_clamp_value)

        with torch.autocast(device_type=cfg.device.type, enabled=False):
            control_pred = self.control(noisy.float(), cond_latents.float(), t.float())

        control_pred = torch.nan_to_num(control_pred.float(), nan=0.0,
                                        posinf=cfg.pred_clamp_value, neginf=-cfg.pred_clamp_value)
        control_pred = torch.tanh(control_pred) * cfg.control_output_scale

        pred     = (base_pred + control_pred).clamp(-cfg.pred_clamp_value, cfg.pred_clamp_value)
        target_v = torch.nan_to_num(target_v, nan=0.0,
                                    posinf=cfg.pred_clamp_value, neginf=-cfg.pred_clamp_value)

        mask = batch["dynamic_mask"].to(device=cfg.device, dtype=torch.float32)  # [B,T,1,H,W]
        mask = mask.permute(0, 2, 1, 3, 4)
        mask = F.interpolate(mask, size=pred.shape[-3:], mode="trilinear", align_corners=False)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        weight = cfg.control_loss_static_weight * (1 - mask) + cfg.control_loss_dynamic_weight * mask

        diff = (pred - target_v).clamp(-cfg.loss_diff_clamp_value, cfg.loss_diff_clamp_value)
        loss = (diff.square() * weight).mean()
        return torch.nan_to_num(loss, nan=0.0,
                                posinf=cfg.loss_diff_clamp_value ** 2, neginf=0.0)

    @torch.no_grad()
    def reconstruct_video(
        self,
        batch: dict,
        t_value: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        cfg     = self.cfg
        batch   = move_batch(batch, cfg.device)
        prompts = batch.get("prompt_text", [cfg.default_prompt] * batch["target"].shape[0])

        latents      = self.encode_video(batch["target"]).float().clamp(-cfg.latent_clamp_value, cfg.latent_clamp_value)
        cond_latents = self.encode_video(self.make_condition_video(batch)).float().clamp(-cfg.latent_clamp_value, cfg.latent_clamp_value)

        B   = latents.shape[0]
        t_s = min(max(float(t_value), cfg.timestep_min), cfg.timestep_max)
        t   = torch.full((B,), t_s, device=cfg.device, dtype=torch.float32)
        ex  = (slice(None),) + (None,) * (latents.ndim - 1)

        noise        = torch.randn_like(latents).clamp(-cfg.noise_clamp_value, cfg.noise_clamp_value)
        noisy        = ((1 - t[ex]) * latents + t[ex] * noise).clamp(-cfg.latent_clamp_value, cfg.latent_clamp_value)
        prompt_embeds = self._text_embeds(prompts)

        base_pred = self._transformer_call(noisy, t.float(), prompt_embeds, grad_enabled=False)
        with torch.autocast(device_type=cfg.device.type, enabled=False):
            control_pred = self.control(noisy.float(), cond_latents.float(), t.float())

        control_pred = torch.tanh(control_pred) * cfg.control_output_scale
        pred_v = (base_pred.float() + control_pred).clamp(-cfg.pred_clamp_value, cfg.pred_clamp_value)
        pred_v = torch.nan_to_num(pred_v, nan=0.0, posinf=cfg.pred_clamp_value, neginf=-cfg.pred_clamp_value)

        latents_hat = (noisy.float() - t[ex] * pred_v).clamp(-cfg.latent_clamp_value, cfg.latent_clamp_value)
        video_hat   = self.decode_video(latents_hat)
        target_vid  = batch["target"].float().clamp(-1, 1)

        if video_hat.shape[-2:] != target_vid.shape[-2:]:
            B2, T2, C2, H2, W2 = target_vid.shape
            target_vid = F.interpolate(
                target_vid.flatten(0, 1),
                size=video_hat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).view(B2, T2, C2, *video_hat.shape[-2:])

        if video_hat.shape[1] != target_vid.shape[1]:
            T2 = min(video_hat.shape[1], target_vid.shape[1])
            video_hat  = video_hat[:, :T2]
            target_vid = target_vid[:, :T2]

        return video_hat, target_vid, batch

    # ------------------------------------------------------------------
    # Export / restore
    # ------------------------------------------------------------------

    def export_trainable_state(self) -> dict:
        control_state = None
        if self.control is not None:
            control_state = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in self.control.state_dict().items()
            }
        return {
            "control":              control_state,
            "lora":                 {
                k: v.detach().cpu()
                for k, v in self.transformer.state_dict().items()
                if "lora" in k.lower()
            },
            "lora_target_modules":  self.lora_target_modules,
            "wan_dir":              str(self.wan_dir),
            "control_width":        self.cfg.control_width,
            "control_depth":        self.cfg.control_depth,
            "control_output_scale": self.cfg.control_output_scale,
            "control_channels":     self.control_channels,
        }
