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

import torch
from pathlib import Path

from configs.config import SpatiaConfig
from pipeline.encode import (
    WanVAE, T5Encoder,
    extract_frames, scene_projection, retrieve_reference_frames,
)


def preprocess_one(
    video_path: Path,
    caption: str,
    out_path: Path,
    vae: WanVAE,
    t5: T5Encoder,
    cfg: SpatiaConfig,
) -> bool:
    """
    Xử lý 1 video → lưu .pt file.
    Returns True nếu thành công.
    """
    # Tổng số frames cần trích
    n_total = cfg.target_frames + cfg.preceding_frames + cfg.max_ref_frames * 2

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

    # 6. Text encode
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
    for i, (vp, tp) in enumerate(todo):
        out_path = proc_dir / (vp.stem + ".pt")
        caption  = f"A scene from a real estate property video"

        try:
            preprocess_one(vp, caption, out_path, vae, t5, cfg)
            ok += 1
        except Exception as e:
            print(f"\n  [WARN] {vp.name}: {e}")

        print(f"  {i+1:3d}/{len(todo)} | ok={ok}", end="\r")

    print(f"\n[Preprocess] Done. {ok}/{len(todo)} files saved.")
