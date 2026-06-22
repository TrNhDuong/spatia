"""
spatia_pipeline/preprocessing/mapanything_module.py
-----------------------------------------------------
MapAnything geometry helper + depth / pose control rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from spatia_pipeline.data.video_utils import intrinsics_to_K


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def fallback_depth_from_frames(frames: np.ndarray) -> np.ndarray:
    """
    Produce a pseudo-depth map from frame texture/blur (no external model).

    Args:
        frames: uint8 [T, H, W, 3]
    Returns:
        depths: float32 [T, H, W] in [0, 1]
    """
    depths: List[np.ndarray] = []
    for fr in frames:
        gray  = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blur  = cv2.GaussianBlur(gray, (0, 0), 3)
        edge  = cv2.Laplacian(blur, cv2.CV_32F)
        depth = _normalize01(1.0 - blur + 0.15 * np.abs(edge))
        depth = cv2.GaussianBlur(depth, (5, 5), 0)
        depths.append(depth.astype(np.float32))
    return np.stack(depths)


# ---------------------------------------------------------------------------
# Rendering helpers (no model needed)
# ---------------------------------------------------------------------------

def render_depth_control(
    depths: np.ndarray,
    masks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Color-map depth maps and optionally overlay dynamic-object masks.

    Args:
        depths: float32 [T, H, W]
        masks:  optional uint8 [T, H, W], values 0 or 255
    Returns:
        uint8 [T, H, W, 3] RGB control frames
    """
    outs: List[np.ndarray] = []
    for i, depth in enumerate(depths):
        d8    = (_normalize01(depth) * 255).astype(np.uint8)
        color = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        if masks is not None:
            m = masks[min(i, len(masks) - 1)] > 0
            color[m] = (
                0.35 * color[m] + np.array([255, 40, 40]) * 0.65
            ).astype(np.uint8)
        outs.append(color)
    return np.stack(outs)


def render_pose_control(
    selected_poses: List[dict],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Render top-down camera trajectory visualisation.

    Args:
        selected_poses: list of pose dicts (each has "pose" 3×4)
        height, width:  output frame dimensions
    Returns:
        uint8 [T, H, W, 3] RGB control frames
    """
    translations = np.array([p["pose"][:, 3] for p in selected_poses], dtype=np.float32)
    x = translations[:, 0]
    z = translations[:, 2]

    if np.ptp(x) < 1e-6:
        x = x + np.linspace(-0.01, 0.01, len(x))
    if np.ptp(z) < 1e-6:
        z = z + np.linspace(-0.01, 0.01, len(z))

    xs = ((x - x.min()) / (x.max() - x.min() + 1e-6) * (width  * 0.8) + width  * 0.1).astype(int)
    ys = ((z - z.min()) / (z.max() - z.min() + 1e-6) * (height * 0.8) + height * 0.1).astype(int)

    controls: List[np.ndarray] = []
    for i in range(len(selected_poses)):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(1, len(xs)):
            cv2.line(img, (xs[j - 1], ys[j - 1]), (xs[j], ys[j]), (40, 80, 160), 1)
        for j in range(1, i + 1):
            cv2.line(img, (xs[j - 1], ys[j - 1]), (xs[j], ys[j]), (60, 220, 80), 2)
        cv2.circle(img, (xs[i], ys[i]), 5, (255, 80, 60), -1)
        controls.append(img)

    return np.stack(controls)


def compose_memory_control(
    depth_control: np.ndarray,
    pose_control: np.ndarray,
) -> np.ndarray:
    """Blend depth and pose control frames (65% depth, 35% pose)."""
    return np.clip(
        0.65 * depth_control.astype(np.float32) + 0.35 * pose_control.astype(np.float32),
        0, 255,
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# MapAnything module
# ---------------------------------------------------------------------------

class MapAnythingModule:
    """
    Lazy wrapper for the MapAnything 3-D perception model.

    Falls back to :func:`fallback_depth_from_frames` when unavailable.
    """

    def __init__(
        self,
        repo: Optional[Path | str],
        model_path: Optional[Path | str],
        device: torch.device,
        enabled: bool = True,
        strict: bool = False,
    ) -> None:
        self.repo       = Path(repo) if repo is not None else None
        self.model_path = Path(model_path) if model_path is not None else None
        self.device     = device
        self.enabled    = enabled
        self.strict     = strict

        self._model  = None
        self._loaded = False
        self._error: Optional[str] = None

    # ------------------------------------------------------------------

    def _load(self) -> Optional[object]:
        if self._loaded:
            return self._model

        self._loaded = True

        if not self.enabled:
            return None

        try:
            if self.repo is not None:
                for p in [self.repo, self.repo.parent]:
                    if str(p) not in sys.path:
                        sys.path.insert(0, str(p))

            from mapanything.models import MapAnything  # type: ignore

            model_id   = str(self.model_path) if self.model_path is not None else "facebook/map-anything"
            local_only = self.model_path is not None
            print("Loading MapAnything from", model_id)

            model = (
                MapAnything.from_pretrained(model_id, local_files_only=local_only)
                .to(self.device)
                .eval()
            )
            self._model = model
            return model

        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            print("WARN MapAnything load failed:", self._error)
            if self.strict:
                raise
            return None

    # ------------------------------------------------------------------

    def get_depth_and_points(
        self,
        frames: np.ndarray,
        pose_rows: List[dict],
    ) -> Tuple[np.ndarray, Optional[List], Dict]:
        """
        Run MapAnything to get per-frame depth maps and 3-D point clouds.

        Returns:
            depths:   float32 [T, H, W] in [0, 1]
            pts3d:    list of point-cloud arrays per frame (or None)
            meta:     dict with source info
        """
        model = self._load()

        if model is None:
            return fallback_depth_from_frames(frames), None, {
                "source": "fallback_depth",
                "error":  self._error,
            }

        try:
            from mapanything.utils.image import preprocess_inputs  # type: ignore

            views: List[dict] = []
            for fr, pose in zip(frames, pose_rows):
                K = intrinsics_to_K(pose["intrinsics"], fr.shape[0], fr.shape[1])
                views.append({
                    "img":              torch.from_numpy(fr).to(self.device),
                    "intrinsics":       torch.from_numpy(K).to(self.device),
                    "camera_poses":     torch.from_numpy(pose["pose4x4"]).to(self.device),
                    "is_metric_scale":  torch.tensor([True], device=self.device),
                })

            processed_views = preprocess_inputs(views)

            with torch.no_grad():
                predictions = model.infer(
                    processed_views,
                    memory_efficient_inference=True,
                    minibatch_size=1,
                    use_amp=(self.device.type == "cuda"),
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                    apply_confidence_mask=False,
                    ignore_calibration_inputs=False,
                    ignore_pose_inputs=False,
                )

            depths: List[np.ndarray] = []
            pts_list: List = []
            for pred in predictions:
                d = pred.get("depth_z", pred.get("depth_along_ray"))
                if torch.is_tensor(d):
                    d = d.detach().float().cpu().numpy()
                d = np.squeeze(d)
                if d.shape[:2] != frames[0].shape[:2]:
                    d = cv2.resize(
                        d, (frames[0].shape[1], frames[0].shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                depths.append(_normalize01(d))

                pts = pred.get("pts3d")
                if torch.is_tensor(pts):
                    pts = pts.detach().float().cpu().numpy()
                pts_list.append(pts)

            return np.stack(depths).astype(np.float32), pts_list, {"source": "mapanything"}

        except Exception as e:
            print("WARN MapAnything inference failed:", type(e).__name__, e)
            if self.strict:
                raise
            return fallback_depth_from_frames(frames), None, {
                "source": "fallback_depth_infer_failed",
                "error":  f"{type(e).__name__}: {e}",
            }

    def unload(self) -> None:
        self._model  = None
        self._loaded = False
