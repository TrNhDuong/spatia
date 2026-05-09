"""
pipeline/encode.py
─────────────────────────────────────────────────────────
Encoder wrappers:
  - WanVAE   : Wan2.2 VAE  (video frames → latent tokens)
  - T5Encoder: T5-XXL      (text caption → embeddings)
  - helpers  : scene_projection, retrieve_refs, flatten_latent
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from configs.config import SpatiaConfig

log = logging.getLogger(__name__)


# ── Wan2.2 VAE ───────────────────────────────────────────────────────────────────────
class WanVAE:
    """
    Wraps Wan2.2 VAE từ diffusers.
    Fallback: identity encoder (random latent) nếu model chưa được cài.
    """

    def __init__(self, model_name: str, device: str):
        self.device   = device
        self.model    = None
        try:
            from diffusers import AutoencoderKLWan
            self.model = AutoencoderKLWan.from_pretrained(
                model_name, subfolder="vae", torch_dtype=torch.float16
            ).to(device)
            self.model.eval()
            print(f"[Encode] Wan2.2 VAE loaded from '{model_name}'")
        except Exception as e:
            print(f"[Encode][WARN] Could not load Wan2.2 VAE: {e}")
            print("         → Falling back to random latents (for testing).")

    @torch.no_grad()
    def encode(self, frames: torch.Tensor,
               cfg: SpatiaConfig) -> torch.Tensor:
        """
        frames : [T, 3, H, W]  in [-1, 1]
        returns: [T'*H'*W', C]  flattened latent
        """
        H_lat = cfg.height  // cfg.spatial_downsample
        W_lat = cfg.width   // cfg.spatial_downsample
        T_lat = max(1, frames.shape[0] // cfg.temporal_downsample)
        C     = cfg.video_latent_dim

        if self.model is None:
            return torch.randn(T_lat * H_lat * W_lat, C)

        x = frames.to(self.device, dtype=torch.float16)
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4)       # [1, C, T, H, W]
        latent = self.model.encode(x).latent_dist.sample()  # [1, C, T', H', W']
        latent = latent.squeeze(0)                          # [C, T', H', W']
        C_out, T_out, H_out, W_out = latent.shape
        out = latent.permute(1, 2, 3, 0).reshape(T_out * H_out * W_out, C_out)
        return out.cpu().float()


# ── T5 Encoder ───────────────────────────────────────────────────────────────────────
class T5Encoder:
    """
    Wraps T5 text encoder từ HuggingFace transformers.
    Fallback: random embeddings nếu model chưa được cài.
    """

    def __init__(self, model_name: str, device: str, max_length: int = 77):
        self.device     = device
        self.max_length = max_length
        self.tokenizer  = None
        self.model      = None
        try:
            from transformers import T5Tokenizer, T5EncoderModel
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model     = T5EncoderModel.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(device)
            self.model.eval()
            print(f"[Encode] T5 loaded from '{model_name}'")
        except Exception as e:
            print(f"[Encode][WARN] Could not load T5: {e}")
            print("         → Falling back to random text tokens (for testing).")

    @torch.no_grad()
    def encode(self, caption: str, text_dim: int) -> torch.Tensor:
        """Returns [max_length, text_dim]"""
        if self.tokenizer is None:
            return torch.randn(self.max_length, text_dim)

        inputs = self.tokenizer(
            caption, return_tensors="pt",
            padding="max_length", truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        out = self.model(input_ids=inputs.input_ids,
                         attention_mask=inputs.attention_mask)
        return out.last_hidden_state.squeeze(0).cpu().float()


# ── Frame extraction ─────────────────────────────────────────────────────────────────────
def extract_frames(video_path: str, num_frames: int,
                   height: int, width: int) -> torch.Tensor:
    """
    Trích num_frames frame đều nhau từ video.
    Returns [num_frames, 3, H, W] in [-1, 1].

    Raises RuntimeError nếu không đọc được video — không silent-catch.
    Cần import cv2 (opencv-python).
    """
    import cv2  # bắt buộc phải có; ImportError sẽ propagate rõ ràng

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 2:
        cap.release()
        raise RuntimeError(
            f"Video '{video_path}' quá ngắn ({total} frames). "
            "Đảm bảo video đã được validate bằng preprocess_one trước khi gọi hàm này."
        )

    indices = [int(i) for i in
               torch.linspace(0, total - 1, num_frames).tolist()]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(
                f"Không đọc được frame {idx} từ '{video_path}'. "
                "Video có thể bị hỏng hoặc bị cắt giữa chừng."
            )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    cap.release()

    arr = torch.from_numpy(np.stack(frames)).float() / 127.5 - 1.0
    return arr.permute(0, 3, 1, 2)   # [T, 3, H, W]


# ── Scene projection (MapAnything placeholder) ────────────────────────────
def scene_projection(frames: torch.Tensor) -> torch.Tensor:
    """
    Placeholder cho MapAnything point-cloud projection.
    Áp dụng edge-fading mask để mô phỏng scene projection.

    Thay thế bằng MapAnything khi available:
        point_cloud = mapanything(frames)
        return render_projection(point_cloud, camera_poses)
    """
    T, C, H, W = frames.shape
    yy = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(T, C, H, W)
    xx = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(T, C, H, W)
    mask = (1 - (xx.abs() + yy.abs()) / 2).clamp(0, 1)
    return frames * mask


# ── Reference frame retrieval (Algorithm 1, paper) ───────────────────────
def retrieve_reference_frames(target: torch.Tensor,
                               candidates: torch.Tensor,
                               K: int) -> torch.Tensor:
    """
    Chọn K candidate frames có spatial overlap cao nhất với target clip.
    Dùng cosine similarity trên mean frame (proxy cho 3D overlap của paper).

    target     : [N, 3, H, W]
    candidates : [O, 3, H, W]
    Returns    : [K, 3, H, W]
    """
    t = F.normalize(target.mean(0).flatten().unsqueeze(0), dim=-1)
    scores = [
        (F.normalize(candidates[i].flatten().unsqueeze(0), dim=-1) * t)
        .sum().item()
        for i in range(len(candidates))
    ]
    top_k = sorted(range(len(candidates)),
                   key=lambda i: scores[i], reverse=True)[:K]
    sel = [candidates[i] for i in top_k]
    while len(sel) < K:
        sel.append(torch.zeros_like(candidates[0]))
    return torch.stack(sel)
