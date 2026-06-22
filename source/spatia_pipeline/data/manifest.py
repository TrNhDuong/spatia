"""
spatia_pipeline/data/manifest.py
----------------------------------
Load and validate the RealEstate10K video-pose manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _find_child_dir(root: Path, name: str) -> Optional[Path]:
    matches = [p for p in root.rglob(name) if p.is_dir()]
    return matches[0] if matches else None


def load_manifest(
    data_root: Path,
    max_videos: int,
    default_prompt: str,
    use_prompt_csv: bool = True,
) -> List[Dict]:
    """
    Scan ``data_root`` for video/pose pairs.

    Returns a list of dicts with keys:
        id, video_path, pose_path, prompt
    """
    data_root = Path(data_root)

    video_dir = _find_child_dir(data_root, "videos") or data_root
    pose_dir  = _find_child_dir(data_root, "poses")  or data_root

    manifest_candidates = list(data_root.rglob("manifest.csv"))
    prompt_candidates   = list(data_root.rglob("prompts.csv"))
    manifest_path = manifest_candidates[0] if manifest_candidates else None
    prompt_csv    = prompt_candidates[0]   if prompt_candidates   else None

    print("VIDEO_DIR:", video_dir)
    print("POSE_DIR:", pose_dir)
    print("MANIFEST_PATH:", manifest_path)
    print("PROMPT_CSV:", prompt_csv)

    video_files = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    pose_files  = sorted(p for p in pose_dir.rglob("*.txt"))
    pose_by_stem: Dict[str, Path] = {p.stem: p for p in pose_files}

    # Optional per-video prompts
    prompt_by_stem: Dict[str, str] = {}
    if use_prompt_csv and prompt_csv is not None:
        pdf = pd.read_csv(prompt_csv)
        for _, row in pdf.iterrows():
            stem = Path(str(row.get("video_path", row.get("id", "")))).stem
            prompt = str(row.get("prompt", default_prompt))
            if stem:
                prompt_by_stem[stem] = prompt

    records: List[Dict] = []

    if manifest_path is not None:
        mdf = pd.read_csv(manifest_path)
        for _, row in mdf.iterrows():
            status = str(row.get("status", "ok"))
            if status not in {"ok", "exists", ""}:
                continue
            vp_raw = str(row.get("video_path", ""))
            pp_raw = str(row.get("pose_path",  ""))
            stem   = str(row.get("id", Path(vp_raw).stem))
            vp = Path(vp_raw)
            pp = Path(pp_raw)
            if not vp.exists():
                cand = [p for p in video_files if p.stem == stem]
                vp = cand[0] if cand else None   # type: ignore[assignment]
            if not pp.exists():
                pp = pose_by_stem.get(stem)      # type: ignore[assignment]
            if vp is not None and pp is not None and vp.exists() and pp.exists():
                records.append({
                    "id": stem,
                    "video_path": str(vp),
                    "pose_path":  str(pp),
                    "prompt":     prompt_by_stem.get(stem, default_prompt),
                })
    else:
        for vp in video_files:
            pp = pose_by_stem.get(vp.stem)
            if pp is not None:
                records.append({
                    "id":         vp.stem,
                    "video_path": str(vp),
                    "pose_path":  str(pp),
                    "prompt":     prompt_by_stem.get(vp.stem, default_prompt),
                })

    records = records[:max_videos]
    print(f"Usable video-pose pairs: {len(records)}")
    for r in records[:5]:
        print(r)

    if len(records) < min(10, max_videos):
        raise RuntimeError(
            "Too few usable video/pose pairs. "
            "Check videos/poses naming and manifest.csv."
        )

    return records
