"""
pipeline/preprocess.py
─────────────────────────────────────────────────────────
Step 3: Video → .pt latent files (offline, chạy 1 lần trước training).

Mỗi file .pt chứa:
    x_T    [N_T, C]          target video latent
    x_P    [N_P, C]          preceding video latent
    x_R    [N_R, C]          K reference frame latents
    x_S_T  [N_T, C]          target scene projection latent
    x_S_P  [N_P, C]          preceding scene projection latent
    text   [N_txt, text_dim] T5 text embedding
"""

import logging
import torch
from pathlib import Path

from configs.config import SpatiaConfig
from pipeline.encode import (
    WanVAE, T5Encoder,
    extract_frames, scene_projection, retrieve_reference_frames,
)

log = logging.getLogger(__name__)

# Minimum frames required so we can fill target + preceding + 2x ref candidate pool
# Matches: target_frames + preceding_frames + max_ref_frames * 2
_MIN_DURATION_SEC = 3.0   # videos shorter than this are skipped


def _build_caption_from_txt(txt_path: "Path | None") -> str:
    """
    Đọc camera pose metadata từ file .txt của RealEstate10K và xây dựng
    caption mô tả chuyển động camera.

    Format thực tế RealEstate10K (19 tokens/dòng):
        ts fx fy cx cy k1 k2 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3
         0  1  2  3  4  5  6   7   8   9  10  11  12  13  14  15 16 17 18
    """
    if txt_path is None or not txt_path.exists():
        return "A scene from a real estate property video"

    def _parse_translation(line: str):
        parts = line.split()
        # t1, t2, t3 nằm ở index 16, 17, 18 (xác minh từ file thực)
        if len(parts) < 19:
            raise ValueError(f"Expected ≥19 cols, got {len(parts)}")
        return float(parts[16]), float(parts[17]), float(parts[18])

    try:
        lines = txt_path.read_text().splitlines()
        pose_lines = [l for l in lines[1:] if l.strip() and not l.startswith('#')]
        if len(pose_lines) < 2:
            return "A scene from a real estate property video"

        t_start = _parse_translation(pose_lines[0])
        t_end   = _parse_translation(pose_lines[-1])

        dx = t_end[0] - t_start[0]
        dy = t_end[1] - t_start[1]
        dz = t_end[2] - t_start[2]

        dominant = max(abs(dx), abs(dy), abs(dz))
        if dominant < 0.05:
            motion = "static camera"
        elif abs(dz) == dominant:
            motion = "camera moving forward" if dz > 0 else "camera moving backward"
        elif abs(dx) == dominant:
            motion = "camera panning right" if dx > 0 else "camera panning left"
        else:
            motion = "camera tilting up" if dy > 0 else "camera tilting down"

        return f"A real estate property scene with {motion}"

    except Exception as exc:
        log.warning("Could not parse camera poses from '%s': %s", txt_path, exc)
        return "A scene from a real estate property video"


def preprocess_one(
    video_path: Path,
    out_path: Path,
    vae: WanVAE,
    t5: T5Encoder,
    cfg: SpatiaConfig,
    txt_path: Path | None = None,
) -> bool:
    """
    Xử lý 1 video → lưu .pt file.
    Returns True nếu thành công.

    Raises ValueError nếu video không đủ frames để tạo đúng clip.
    """
    # Tổng số frames cần trích
    n_total = cfg.target_frames + cfg.preceding_frames + cfg.max_ref_frames * 2

    # ── Bug #1 Fix: Validate clip duration TRƯỚC khi extract ──────────────
    try:
        import cv2
        cap   = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        duration_sec = total_frames / fps
    except Exception as e:
        raise ValueError(f"Không đọc được metadata video: {e}")

    if duration_sec < _MIN_DURATION_SEC:
        raise ValueError(
            f"Clip quá ngắn ({duration_sec:.1f}s < {_MIN_DURATION_SEC}s). "
            f"Cần ít nhất {n_total} frames ({n_total/fps:.1f}s @ {fps:.0f}fps)."
        )
    if total_frames < n_total:
        raise ValueError(
            f"Clip chỉ có {total_frames} frames, cần {n_total}."
        )
    # ──────────────────────────────────────────────────────────────────────

    # 1. Trích frames
    all_frames = extract_frames(str(video_path), n_total,
                                cfg.height, cfg.width)

    # 2. Chia thành target / preceding / candidate
    n_T = cfg.target_frames
    n_P = cfg.preceding_frames

    T_frames = all_frames[:n_T]
    P_frames = all_frames[n_T: n_T + n_P]
    C_frames = all_frames[n_T + n_P:]

    # 3. Retrieve K reference frames từ candidate set
    R_frames = retrieve_reference_frames(T_frames, C_frames, cfg.max_ref_frames)

    # 4. Scene projection (MapAnything placeholder)
    S_T = scene_projection(T_frames)
    S_P = scene_projection(P_frames)

    # 5. VAE encode → flattened latents
    x_T   = vae.encode(T_frames, cfg)
    x_P   = vae.encode(P_frames, cfg)
    x_R   = vae.encode(R_frames, cfg)
    x_S_T = vae.encode(S_T,     cfg)
    x_S_P = vae.encode(S_P,     cfg)

    # 6. Build caption từ camera pose metadata (Bug #4/#5 Fix)
    caption = _build_caption_from_txt(txt_path)
    text = t5.encode(caption, cfg.text_dim)

    torch.save({
        "x_T":   x_T,
        "x_P":   x_P,
        "x_R":   x_R,
        "x_S_T": x_S_T,
        "x_S_P": x_S_P,
        "text":  text,
    }, out_path)
    return True


def preprocess_all(
    video_list: list[tuple[Path, Path]],
    proc_dir: Path,
    cfg: SpatiaConfig,
    device: str,
    vae_name: str,
    t5_name: str,
) -> None:
    """
    Xử lý toàn bộ danh sách video.

    video_list : list of (video_path, txt_path)
    proc_dir   : thư mục output chứa .pt files
    """
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Bỏ qua các file đã xử lý
    done  = {p.stem for p in proc_dir.glob("*.pt")}
    todo  = [(vp, tp) for vp, tp in video_list if vp.stem not in done]

    if not todo:
        n = len(list(proc_dir.glob("*.pt")))
        print(f"[Preprocess] All {n} files already done. Skipping.")
        return

    print(f"[Preprocess] Processing {len(todo)} videos → {proc_dir}")

    # Load encoders (tự động fallback nếu chưa cài)
    vae = WanVAE(vae_name, device)
    t5  = T5Encoder(t5_name, device)

    ok = 0
    skipped = 0
    for i, (vp, tp) in enumerate(todo):
        out_path = proc_dir / (vp.stem + ".pt")

        try:
            preprocess_one(vp, out_path, vae, t5, cfg, txt_path=tp)
            ok += 1
        except ValueError as e:
            # Bug #1 Fix: skip video quá ngắn hoặc thiếu frames (expected)
            log.warning("Skipped '%s': %s", vp.name, e)
            skipped += 1
        except Exception as e:
            log.error("Error processing '%s': %s", vp.name, e, exc_info=True)

        print(f"  {i+1:3d}/{len(todo)} | ok={ok} skipped={skipped}", end="\r")

    print(f"\n[Preprocess] Done. {ok}/{len(todo)} files saved, {skipped} skipped.")
