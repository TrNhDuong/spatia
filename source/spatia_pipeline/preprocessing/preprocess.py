"""
spatia_pipeline/preprocessing/preprocess.py
---------------------------------------------
Main offline preprocessing loop: video → cached .pt sample.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List

import torch
from tqdm.auto import tqdm

from spatia_pipeline.config import SpatiaConfig
from spatia_pipeline.data.video_utils import (
    load_clip_from_video,
    read_pose_file,
    to_tensor_mask,
    to_tensor_video,
)
from spatia_pipeline.preprocessing.keye_module import KeyeModule
from spatia_pipeline.preprocessing.mapanything_module import (
    MapAnythingModule,
    compose_memory_control,
    render_depth_control,
    render_pose_control,
)
from spatia_pipeline.preprocessing.referdino_module import ReferDinoModule
from spatia_pipeline.preprocessing.reference_selection import select_reference_frames


def run_preprocess(
    records: List[dict],
    cfg: SpatiaConfig,
    keye: KeyeModule,
    referdino: ReferDinoModule,
    mapanything: MapAnythingModule,
    proc_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> List[Path]:
    """
    Preprocess all video records and cache results to ``proc_dir`` as ``.pt`` files.

    Args:
        records:       list of dicts {id, video_path, pose_path, prompt}
        cfg:           pipeline configuration
        keye:          KeyeModule instance
        referdino:     ReferDinoModule instance
        mapanything:   MapAnythingModule instance
        proc_dir:      output directory (defaults to cfg.proc_dir)
        skip_existing: if True, skip records whose .pt file already exists

    Returns:
        List of paths to successfully written ``.pt`` files.
    """
    from typing import Optional  # local import avoids circular at module level

    if proc_dir is None:
        proc_dir = cfg.proc_dir
    proc_dir = Path(proc_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    for rec in tqdm(records, desc="Full preprocess"):
        out_path = proc_dir / f"{rec['id']}.pt"
        if skip_existing and out_path.exists():
            written.append(out_path)
            continue

        try:
            _, pose_rows = read_pose_file(rec["pose_path"])
            frames, selected_poses = load_clip_from_video(
                rec["video_path"], pose_rows,
                cfg.total_sample_frames, cfg.height, cfg.width,
            )

            # Split into candidate / prev / target segments
            c  = cfg.candidate_frames
            p  = cfg.prev_frames

            candidate_frames = frames[:c]
            prev_frames      = frames[c:c + p]
            target_frames    = frames[c + p:]

            candidate_poses  = selected_poses[:c]
            prev_poses       = selected_poses[c:c + p]
            target_poses     = selected_poses[c + p:]

            # --- Keye-VL description ---
            keye_meta   = keye.describe_video(rec["video_path"], rec["prompt"])
            entity_text = ", ".join(keye_meta.get("entities", ["moving objects"]))

            # --- Dynamic masks ---
            masks_all, mask_meta = referdino.get_masks(frames, entity_text)
            target_masks = masks_all[c + p:]

            # --- Depth + pose control ---
            depth_all, pts3d, map_meta = mapanything.get_depth_and_points(frames, selected_poses)
            target_depth   = depth_all[c + p:]
            depth_control  = render_depth_control(target_depth, target_masks)
            pose_control   = render_pose_control(target_poses, cfg.height, cfg.width)
            memory_control = compose_memory_control(depth_control, pose_control)

            # --- Reference frame selection ---
            refs, ref_indices = select_reference_frames(
                candidate_frames, candidate_poses, target_poses,
                ref_frames=cfg.ref_frames,
                height=cfg.height,
                width=cfg.width,
            )

            sample = {
                "id":           rec["id"],
                "prev":         to_tensor_video(prev_frames),
                "target":       to_tensor_video(target_frames),
                "control":      to_tensor_video(depth_control),
                "memory":       to_tensor_video(memory_control),
                "dynamic_mask": to_tensor_mask(target_masks),
                "reference":    to_tensor_video(refs),
                "prompt":       keye_meta.get("prompt", rec["prompt"]),
                "entities":     keye_meta.get("entities", ["moving objects"]),
                "video_path":   rec["video_path"],
                "pose_path":    rec["pose_path"],
                "module_meta": {
                    "keye":              keye_meta,
                    "referdino":         mask_meta,
                    "mapanything":       map_meta,
                    "reference_indices": ref_indices,
                },
            }
            torch.save(sample, out_path)
            written.append(out_path)

        except Exception as e:
            print(f"WARN preprocess failed: {rec['id']} {type(e).__name__}: {e}")
            if cfg.strict_external_models:
                raise

    # Free model memory after preprocessing
    keye.unload()
    mapanything.unload()
    referdino.unload()
    gc.collect()
    if cfg.device.type == "cuda":
        torch.cuda.empty_cache()

    processed = sorted(proc_dir.glob("*.pt"))
    print(f"Processed samples: {len(processed)}")

    if len(processed) < 5:
        raise RuntimeError(
            "Too few processed samples. "
            "Check video/pose files and module errors above."
        )

    return processed
