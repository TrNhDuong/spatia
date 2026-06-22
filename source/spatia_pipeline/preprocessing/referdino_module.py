"""
spatia_pipeline/preprocessing/referdino_module.py
---------------------------------------------------
ReferDINO dynamic-object mask helper + optical-flow fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Optical-flow fallback (no external models needed)
# ---------------------------------------------------------------------------

def optical_flow_motion_masks(
    frames: np.ndarray,
    threshold_percentile: int = 85,
) -> np.ndarray:
    """
    Compute per-frame motion masks via dense optical flow (Farneback).

    Args:
        frames: uint8 array [T, H, W, 3]
        threshold_percentile: motion magnitude percentile above which pixels are "moving"

    Returns:
        masks: uint8 array [T, H, W], values 0 or 255
    """
    masks: List[np.ndarray] = []
    prev_gray: Optional[np.ndarray] = None

    for fr in frames:
        gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        if prev_gray is None:
            masks.append(np.zeros(gray.shape, dtype=np.uint8))
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag  = np.hypot(flow[..., 0], flow[..., 1])
            thr  = np.percentile(mag, threshold_percentile)
            mask = (mag > max(float(thr), 0.15)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
            masks.append(mask)
        prev_gray = gray

    return np.stack(masks)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml_config(path: Path) -> dict:
    try:
        from ruamel.yaml import YAML  # type: ignore
        yaml = YAML(typ="safe", pure=True)
        with open(path, encoding="utf-8") as f:
            return yaml.load(f)
    except Exception:
        import yaml as _yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return _yaml.safe_load(f)


def _make_easydict(data: dict) -> object:
    try:
        from easydict import EasyDict  # type: ignore
        return EasyDict(data)
    except Exception:
        from spatia_pipeline.preprocessing._attr_dict import AttrDict
        return AttrDict(data)


# ---------------------------------------------------------------------------
# ReferDINO module
# ---------------------------------------------------------------------------

class ReferDinoModule:
    """
    Lazy wrapper for the ReferDINO video grounding model.

    Falls back to :func:`optical_flow_motion_masks` when:
    - ``enabled=False``
    - the model fails to load
    - inference raises an exception
    """

    def __init__(
        self,
        repo: Optional[Path | str],
        ckpt: Optional[Path | str],
        device: torch.device,
        enabled: bool = True,
        strict: bool = False,
    ) -> None:
        self.repo    = Path(repo) if repo is not None else None
        self.ckpt    = Path(ckpt) if ckpt is not None else None
        self.device  = device
        self.enabled = enabled
        self.strict  = strict

        self._model  = None
        self._args   = None
        self._loaded = False
        self._error: Optional[str] = None

    # ------------------------------------------------------------------

    def _add_paths(self) -> None:
        if self.repo is None:
            return
        for p in [
            self.repo,
            self.repo / "models" / "GroundingDINO",
            self.repo / "models" / "GroundingDINO" / "ops",
        ]:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

    def _try_bootstrap(self) -> None:
        """Run kaggle_bootstrap_referdino if available."""
        if self.repo is None:
            return
        bootstrap_py = self.repo / "kaggle_bootstrap_referdino.py"
        if not bootstrap_py.exists():
            return
        try:
            sys.path.insert(0, str(self.repo))
            from kaggle_bootstrap_referdino import bootstrap  # type: ignore
            built = bootstrap(self.repo, jobs=2, verbose=True)
            self.repo = Path(built)
            self._add_paths()
        except Exception as e:
            print("WARN late ReferDINO bootstrap failed:", type(e).__name__, e)
            if self.strict:
                raise

    def _load(self) -> Tuple[Optional[object], Optional[object]]:
        if self._loaded:
            return self._model, self._args

        self._loaded = True

        if not self.enabled:
            self._error = "RUN_REFERDINO=False"
            return None, None

        if self.repo is None or self.ckpt is None:
            self._error = "ReferDINO repo or checkpoint not found"
            return None, None

        try:
            self._add_paths()
            self._try_bootstrap()

            # Locate YAML config
            config_path = self.repo / "configs" / "ytvos_swinb.yaml"
            if not config_path.exists():
                yamls = sorted(self.repo.rglob("*.yaml"))
                if not yamls:
                    raise FileNotFoundError("No ReferDINO yaml config found")
                config_path = yamls[0]

            config = _load_yaml_config(config_path)
            config = {k: v.get("value", v) if isinstance(v, dict) else v for k, v in config.items()}
            args   = _make_easydict({
                **config,
                "checkpoint_path": str(self.ckpt),
                "device":          "cuda" if self.device.type == "cuda" else "cpu",
                "enable_amp":      self.device.type == "cuda",
                "tracking_alpha":  0.1,
            })
            if hasattr(args, "GroundingDINO"):
                args.GroundingDINO.tracking_alpha = args.tracking_alpha  # type: ignore[attr-defined]

            from models import build_model  # type: ignore  # ReferDINO repo on sys.path
            model, _, _ = build_model(args)
            model.to(args.device)

            checkpoint = torch.load(str(self.ckpt), map_location="cpu")
            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get("model", checkpoint),
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"ReferDINO loaded. missing={len(missing)}, unexpected={len(unexpected)}")
            model.eval()

            self._model = model
            self._args  = args
            return model, args

        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            print("WARN ReferDINO load failed:", self._error)
            if self.strict:
                raise
            return None, None

    # ------------------------------------------------------------------

    def get_masks(
        self,
        frames: np.ndarray,
        expression: str,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Produce per-frame binary masks for the given textual expression.

        Returns:
            masks:  uint8 array [T, H, W], values 0 or 255
            meta:   dict with source info
        """
        model, args = self._load()

        if model is None:
            return optical_flow_motion_masks(frames), {
                "source": "optical_flow_load_failed",
                "error":  self._error,
            }

        try:
            import torchvision.transforms as T  # type: ignore
            from misc import nested_tensor_from_videos_list  # type: ignore  # ReferDINO

            transform = T.Compose([
                T.Resize(360),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            pil_frames = [Image.fromarray(fr) for fr in frames]
            imgs       = torch.stack([transform(img) for img in pil_frames], dim=0).to(args.device)
            samples    = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)

            img_h, img_w = imgs.shape[-2:]
            target = {"size": torch.as_tensor([int(img_h), int(img_w)]).to(args.device)}
            exp    = " ".join(str(expression).lower().split())

            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=(args.device == "cuda")):
                    outputs = model.infer(samples, [exp], [target])

            pred_logits = outputs["pred_logits"][0]
            pred_masks  = outputs["pred_masks"][0]
            pred_scores = pred_logits.sigmoid().mean(0)
            max_scores, _ = pred_scores.max(-1)
            max_ind    = max_scores.argmax(-1)
            video_len  = len(frames)
            max_inds   = max_ind.repeat(video_len)

            pred_masks = pred_masks[range(video_len), max_inds, ...].unsqueeze(0)
            pred_masks = pred_masks[:, :, :img_h, :img_w].cpu()

            origin_h, origin_w = frames[0].shape[:2]
            pred_masks = F.interpolate(
                pred_masks, size=(origin_h, origin_w), mode="bilinear", align_corners=False
            )
            masks = (pred_masks.sigmoid() > 0.5).squeeze(0).numpy().astype(np.uint8) * 255
            return masks, {"source": "referdino", "expression": exp}

        except Exception as e:
            print("WARN ReferDINO inference failed:", type(e).__name__, e)
            if self.strict:
                raise
            return optical_flow_motion_masks(frames), {
                "source": "optical_flow_infer_failed",
                "error":  f"{type(e).__name__}: {e}",
            }

    def unload(self) -> None:
        self._model  = None
        self._loaded = False
