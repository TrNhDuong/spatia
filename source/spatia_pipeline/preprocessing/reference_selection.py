"""
spatia_pipeline/preprocessing/reference_selection.py
------------------------------------------------------
Reference frame selection based on camera overlap score.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def camera_overlap_score(pose_a: dict, pose_b: dict) -> float:
    """
    Score the visual overlap between two camera poses.

    Higher is better (closer translation + similar rotation).
    """
    ta = pose_a["pose"][:, 3]
    tb = pose_b["pose"][:, 3]
    dist    = float(np.linalg.norm(ta - tb))
    ra      = pose_a["pose"][:, :3]
    rb      = pose_b["pose"][:, :3]
    rot_sim = float(np.trace(ra.T @ rb)) / 3.0
    return -dist + 0.1 * rot_sim


def select_reference_frames(
    candidate_frames: np.ndarray,
    candidate_poses: List[dict],
    target_poses: List[dict],
    ref_frames: int = 7,
    height: int = 192,
    width: int = 320,
) -> Tuple[np.ndarray, List[int]]:
    """
    Select ``ref_frames`` candidate frames that best overlap with the target views.

    Returns:
        refs:     uint8 [ref_frames, H, W, 3]
        selected: list of selected candidate indices
    """
    if len(candidate_frames) == 0:
        return np.zeros((ref_frames, height, width, 3), dtype=np.uint8), []

    scores: List[Tuple[float, int]] = []
    for i, cp in enumerate(candidate_poses):
        s = max(camera_overlap_score(cp, tp) for tp in target_poses)
        scores.append((s, i))

    selected = [i for _, i in sorted(scores, reverse=True)[:ref_frames]]
    refs = [candidate_frames[i] for i in selected]

    # Pad with last frame if too few candidates
    while len(refs) < ref_frames:
        refs.append(refs[-1])

    return np.stack(refs), selected
