# Spatia — Claude Context File

> Cập nhật lần cuối: 2026-05-09  
> Dùng file này để nạp context nhanh cho Claude trong session mới.

---

## 1. Project Overview

**Spatia** là pipeline training video generation model dựa trên paper:  
*"Spatia: Video Generation with Updatable Spatial Memory"* — xem `spatia.md` để đọc toàn văn.

**Mục tiêu:** Train model sinh video với spatial memory (3D point cloud) để giữ consistency dài hạn.  
**Backbone:** Wan2.2 (5B params, paper) — hiện implement phiên bản nhỏ hơn (dim=1024) cho local training.  
**Dataset:** RealEstate10K test split (~100 videos) cho dev/test, full split (40K) cho production.

---

## 2. Cấu trúc Project

```
Spatia/
├── configs/config.py          # SpatiaConfig dataclass — tất cả hyperparams
├── models/
│   ├── spatia.py              # Model chính: Spatia(nn.Module)
│   ├── blocks.py              # SpatiaNetworkBlock, MainBlock, ControlNetBlock, FFN
│   ├── attention.py           # MultiHeadAttention (Flash Attention)
│   └── lora.py                # LoRALinear (base weight frozen)
├── training/
│   ├── loss.py                # flow_matching_loss(), logit_normal_sample()
│   └── trainer.py             # train_one_epoch()
├── data/dataset.py            # SpatiaDataset (real .pt mode + dummy mode)
├── pipeline/
│   ├── download.py            # download_metadata(), download_videos()
│   ├── preprocess.py          # preprocess_one(), preprocess_all()
│   └── encode.py              # WanVAE, T5Encoder, extract_frames(), scene_projection()
├── utils/checkpoint.py        # save_checkpoint(), load_checkpoint()
├── run.py                     # Orchestration: download → preprocess → train
├── spatia.md                  # Toàn văn paper (dùng thay PDF)
└── requirements.txt           # Dependencies
```

---

## 3. Key Hyperparameters (SpatiaConfig — khớp với paper)

| Param | Value | Ghi chú |
|---|---|---|
| `dim` | 1024 | Paper dùng ~5B, local dùng 1024 |
| `num_main_blocks` | 8 | 8 SpatiaNetworkBlock |
| `num_sub_blocks` | 4 | 4 MainBlock per network block |
| `num_heads` | 16 | head_dim = 64 |
| `text_dim` | 4096 | T5-XXL output dim |
| `video_latent_dim` | 16 | Wan2.2 VAE latent C |
| `max_ref_frames` | **7** | K=7 (paper Table 5: best) |
| `target_frames` | **81** | 1st iteration (paper) |
| `preceding_frames` | **9** | "conditioned on 9 prev. frames" (paper) |
| `height / width` | 480 / 640 | Paper: 720P |
| `spatial_downsample` | 16 | h_lat=30, w_lat=40 |
| `temporal_downsample` | 4 | t_T=20, t_P=2 slices |
| `stage1_iters` | **8000** | ControlNet training (paper) |
| `stage2_iters` | **5000** | LoRA fine-tune (paper) |
| `lr_controlnet` | **1e-5** | Stage 1 LR (paper) |
| `lr_lora` | **1e-4** | Stage 2 LR (paper) |
| `lora_rank` | **64** | (paper) |
| `aug_t_max` | **50/1000=0.05** | t_aug ∈ [0, 50] normalized |
| `grad_clip` | 1.0 | |
| `batch_size` | 2 | Paper: 64 on 64 GPUs |

---

## 4. Token Dimensions (quan trọng cho debug)

```
h_lat = 480 // 16 = 30
w_lat = 640 // 16 = 40

N_T = (81 // 4) × 30 × 40 = 20 × 30 × 40 = 24,000   # target tokens
N_P = (9  // 4) × 30 × 40 =  2 × 30 × 40 =  2,400   # preceding tokens
N_R = 7         × 30 × 40 =          8,400            # reference tokens
N_txt = 77                                             # text tokens

x_tokens     shape: [B, N_R+N_P+N_T, dim] = [B, 34,800, 1024]
scene_tokens shape: [B, N_P+N_T,     dim] = [B, 26,400, 1024]
velocity out shape: [B, N_T, C]           = [B, 24,000,    16]
```

---

## 5. Bugs Đã Fix (toàn bộ history)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `preprocess.py` | Column index T1,T2,T3 sai (19→**16,17,18**) + broken indentation | ✅ |
| 2 | `models/spatia.py` | `SpatiaNetworkBlock.forward()` thiếu arg `n_T` | ✅ |
| 3 | `models/blocks.py` | Scene token split dùng `// 2` sai khi N_P ≠ N_T | ✅ → dùng `n_P`, `n_T` |
| 4 | `training/trainer.py` | `scheduler.step()` per-epoch → phải per-step | ✅ |
| 5 | `training/trainer.py` | `_LRScheduler` deprecated PyTorch ≥ 2.4 | ✅ → try/except compat |
| 6 | `run.py` | `persistent_workers` crash trên Windows khi workers=0 | ✅ → guard |
| 7 | `pipeline/download.py` | `MIN_CLIP_SEC` quá nhỏ → clip thiếu frames | ✅ → 5.0s |
| 8 | `data/dataset.py` | Không validate `text_dim` của real .pt files | ✅ → assert |
| 9 | `models/attention.py` | Naive O(N²) attention → ~60GB VRAM với 34800 tokens | ✅ → Flash Attention (`scaled_dot_product_attention`) |
| 10 | `models/lora.py` | Base weight không bị frozen | ✅ → `linear.weight.requires_grad_(False)` |
| 11 | `models/spatia.py` | `unfreeze_main_blocks()` unlock base weight LoRA | ✅ → chỉ unfreeze `lora_A`, `lora_B` |
| 12 | `configs/config.py` | `preceding_frames=8` sai → paper nói **9** | ✅ → 9 |

---

## 6. Pipeline Flow (end-to-end)

```
Step 1: download_metadata()
        → tải test.zip từ Google Storage
        → extract → data/raw/realestate/test/*.txt

Step 2: download_videos()
        → parse_txt(): URL, timestamps (µs / 1_000_000 = giây)
        → yt-dlp + ffmpeg: clip [ts[0], ts[-1]], MIN_CLIP_SEC=5.0
        → parallel ThreadPoolExecutor, retry 3x, resume

Step 3: preprocess_one() cho mỗi video
        → validate duration ≥ 3s, total_frames ≥ 104
        → extract_frames(104) → [104, 3, 480, 640]
        → T_frames = [:81], P_frames = [81:90], C_frames = [90:]
        → retrieve_reference_frames(T, C, K=7) → [7, 3, H, W]
        → scene_projection(T), scene_projection(P)  ← MapAnything placeholder
        → vae.encode() → x_T[24000,16], x_P[2400,16], x_R[8400,16]
        → t5.encode(caption) → text[77, 4096]
        → torch.save({x_T, x_P, x_R, x_S_T, x_S_P, text}) → .pt

Step 4: Training
  Stage 1 (ControlNet):
        → model.freeze_main_blocks()
        → AdamW(lr=1e-5), CosineAnnealingLR(T_max=8000)
        → train_one_epoch(..., max_iters=8000)
        → save spatia_stage1.pt

  Stage 2 (LoRA):
        → model.enable_lora()  ← swap MainBlocks, load_state_dict(strict=False)
        → model.to(device)     ← LoRA weights mới tạo trên CPU
        → model.freeze_controlnet()
        → model.unfreeze_main_blocks()  ← CHỈ lora_A, lora_B
        → AdamW(lr=1e-4), CosineAnnealingLR(T_max=5000)
        → train_one_epoch(..., max_iters=5000)
        → save spatia_final.pt
```

---

## 7. Loss Function

```python
# Flow Matching (paper Section 3.1.3):
x_0 ~ N(0, I)
x_t = (1-t)*x_0 + t*x_T          # linear interpolation
u_t = x_T - x_0                   # ground-truth velocity
t   ~ logit_normal(0, 1)          # t = sigmoid(N(0,1))

# Preceding-frame augmentation (paper Section 6):
t_aug ~ Uniform[0, 0.05]
x_P_aug = (1-t_aug)*x_P + t_aug*ε

# Model predicts:
v_t = model(x_t, x_P_aug, x_R, x_S_T, x_S_P, text, t)

# Loss:
L = MSE(v_t, u_t)
```

---

## 8. Architecture (theo paper spatia.md)

```
Spatia = 8 × SpatiaNetworkBlock
         ├── 1 ControlNetBlock (scene tokens)
         │     Self-Attn → Cross-Attn(text) → FFN → Projector(MLP)
         │     Output: scene_out + conditioning signal
         └── 4 × MainBlock (video tokens)
               Self-Attn → Cross-Attn(text) → FFN
               Block[0] nhận additive cond từ ControlNet

Input tokens (concatenated):
  x_tokens     = [X_R | X_P | X_T]   [B, 8400+2400+24000, 1024]
  scene_tokens = [X_S_P | X_S_T]     [B, 2400+24000, 1024]
  text_tokens                         [B, 77, 4096]

ControlNet inject:
  cond_P → x_P slice (x_tokens[:, 8400:10800, :])
  cond_T → x_T slice (x_tokens[:, 10800:, :])
  chỉ inject vào MainBlock đầu tiên (i==0)

Output:
  velocity = out_proj(out_norm(x_tokens[:, 10800:, :]))  → [B, 24000, 16]
```

---

## 9. Lệnh Chạy

```bash
# Test nhanh (dummy data, không cần GPU mạnh)
python run.py --skip_download --skip_preprocess --device cpu --batch_size 1

# Test với real data (5 videos)
python run.py --max_videos 5 --device cuda

# Full run
python run.py --max_videos 100 --device cuda

# Nếu đã có video rồi
python run.py --skip_download --device cuda

# Nếu đã có .pt files rồi
python run.py --skip_download --skip_preprocess --device cuda
```

---

## 10. Dependencies

```
torch>=2.1.0        # F.scaled_dot_product_attention cần >= 2.0
einops>=0.7.0       # dùng trong attention.py
opencv-python       # extract_frames()
transformers        # T5Tokenizer, T5EncoderModel
diffusers           # AutoencoderKLWan (Wan2.2 VAE)
yt-dlp              # download YouTube clips
sentencepiece       # T5Tokenizer cần (critical!)
scipy               # diffusers dependency
tqdm                # progress bar
torchvision         # optional image ops
```

---

## 11. Known Limitations (intentional, không phải bug)

| Item | Lý do |
|------|-------|
| `scene_projection()` dùng edge-fading mask | MapAnything chưa open-source |
| `retrieve_reference_frames()` dùng cosine sim | 3D IoU cần MapAnything |
| `dim=1024` thay vì 5B | Local GPU constraint |
| `height=480` thay vì 720P | Local GPU/VRAM constraint |
| batch_size=2 thay vì 64 | Paper dùng 64×AMD MI250 |

---

## 12. Files Không Cần Nữa

- `2512.15716v1.pdf` — **không cần**, nội dung đã có đầy đủ trong `spatia.md`
