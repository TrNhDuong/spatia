"""
spatia_pipeline/data/video_utils.py
-------------------------------------
Low-level video / pose utilities used by the preprocess pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Pose file
# ---------------------------------------------------------------------------

def read_pose_file(path: str | Path) -> Tuple[str, List[dict]]:
    """
    Parse a RealEstate10K-style pose text file.

    Returns:
        url:  first line (scene URL)
        rows: list of dicts with keys timestamp, intrinsics, pose (3×4),
              pose4x4 (4×4 float32), raw_pose.
    """
    lines = [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    ]
    if len(lines) < 2:
        raise ValueError(f"Pose file too short: {path}")

    url = lines[0]
    rows: List[dict] = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 17:
            continue
        ts    = int(float(parts[0]))
        intr  = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
        pvals = np.array([float(x) for x in parts[5:]], dtype=np.float32)
        pose_3x4 = pvals[:12].reshape(3, 4) if len(pvals) >= 12 else np.zeros((3, 4), np.float32)
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, :4] = pose_3x4
        rows.append({
            "timestamp":  ts,
            "intrinsics": intr,
            "pose":       pose_3x4,
            "pose4x4":    pose_4x4,
            "raw_pose":   pvals,
        })

    if not rows:
        raise ValueError(f"No valid pose rows: {path}")
    return url, rows


# ---------------------------------------------------------------------------
# Video reading
# ---------------------------------------------------------------------------

def read_video_frame_at(
    cap: cv2.VideoCapture,
    sec: float,
    height: int,
    width: int,
) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sec) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def load_clip_from_video(
    video_path: str | Path,
    pose_rows: List[dict],
    total_frames: int,
    height: int,
    width: int,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Extract ``total_frames`` uniformly-spaced frames from *video_path*,
    aligned to ``pose_rows``.

    Returns:
        frames:         uint8 array [T, H, W, 3]
        selected_poses: list of pose dicts (length T)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if len(pose_rows) >= total_frames:
        idxs = np.linspace(0, len(pose_rows) - 1, total_frames).round().astype(int)
        secs = [pose_rows[i]["timestamp"] / 1_000_000.0 for i in idxs]
        selected_poses = [pose_rows[i] for i in idxs]
    else:
        fps          = cap.get(cv2.CAP_PROP_FPS) or 24
        frame_count  = cap.get(cv2.CAP_PROP_FRAME_COUNT) or total_frames
        duration     = frame_count / fps
        secs         = np.linspace(0, max(0.0, duration - 1e-3), total_frames).tolist()
        idxs         = np.linspace(0, len(pose_rows) - 1, total_frames).round().astype(int)
        selected_poses = [pose_rows[i] for i in idxs]

    frames: List[np.ndarray] = []
    last: np.ndarray | None = None
    for sec in secs:
        fr = read_video_frame_at(cap, sec, height, width)
        if fr is None:
            fr = last if last is not None else np.zeros((height, width, 3), dtype=np.uint8)
        frames.append(fr)
        last = fr

    cap.release()
    return np.stack(frames), selected_poses


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def to_tensor_video(frames: np.ndarray) -> torch.Tensor:
    """uint8 [T,H,W,3] → float32 tensor [T,C,H,W] in range [-1,1]."""
    arr = frames.astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


def to_tensor_mask(masks: np.ndarray) -> torch.Tensor:
    """uint8/bool [T,H,W] → float32 tensor [T,1,H,W] in range [0,1]."""
    arr = masks.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = arr[:, None]
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def intrinsics_to_K(intr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert RealEstate10K intrinsics (possibly normalised) to a 3×3 K matrix."""
    fx, fy, cx, cy = [float(x) for x in intr]
    if fx <= 10:
        fx *= width
    if fy <= 10:
        fy *= height
    if cx <= 2:
        cx *= width
    if cy <= 2:
        cy *= height
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
