# Spatia — Training from Scratch on RealEstate10K Test Set

> Implementation of **"Spatia: Video Generation with Updatable Spatial Memory"**  
> Paper: [arXiv 2512.15716](https://arxiv.org/abs/2512.15716)

---

## Project Structure

```
Spatia/
├── run.py                       # ← Entry point duy nhất
├── requirements.txt
├── README.md
│
├── configs/
│   └── config.py                # Toàn bộ hyperparameters (SpatiaConfig)
│
├── pipeline/                    # Pipeline tiền xử lý (chạy trước training)
│   ├── __init__.py
│   ├── download.py              # Step 1-2: tải metadata + video clips
│   ├── encode.py                # WanVAE, T5Encoder, scene projection, ref retrieval
│   └── preprocess.py            # Step 3: video → .pt latent files
│
├── models/                      # Kiến trúc Spatia
│   ├── __init__.py
│   ├── lora.py                  # LoRALinear
│   ├── attention.py             # MultiHeadAttention
│   ├── blocks.py                # FFN, MainBlock, ControlNetBlock, SpatiaNetworkBlock
│   └── spatia.py                # Spatia — full model
│
├── data/
│   ├── __init__.py
│   └── dataset.py               # SpatiaDataset (real mode + dummy mode)
│
├── training/
│   ├── __init__.py
│   ├── loss.py                  # flow_matching_loss, logit_normal_sample
│   └── trainer.py               # train_one_epoch
│
└── utils/
    ├── __init__.py
    └── checkpoint.py            # save_checkpoint, load_checkpoint
```

---

## Cách chạy (1 lệnh)

```bash
python run.py
```

Script tự động thực hiện 4 bước:

| Bước | Mô tả | Có thể bỏ qua |
|------|-------|---------------|
| **Step 1** | Tải metadata RealEstate10K test split | `--skip_download` |
| **Step 2** | Tải 100 video clips qua yt-dlp | `--skip_download` |
| **Step 3** | Encode video → `.pt` latent files | `--skip_preprocess` |
| **Step 4** | Train Spatia 1 epoch (Stage 1 + Stage 2) | — |

---

## Các tùy chọn phổ biến

```bash
# Test nhanh với 10 videos
python run.py --max_videos 10

# Đã có video, bỏ qua download
python run.py --skip_download

# Đã có .pt files, train luôn
python run.py --skip_preprocess

# Dùng CPU (không có GPU)
python run.py --device cpu

# Giảm batch size nếu VRAM thấp
python run.py --batch_size 1

# Dùng T5-base thay T5-XXL (nhẹ hơn ~15x)
python run.py --t5_name "google/t5-v1_1-base"

# Tất cả tùy chọn
python run.py \
    --max_videos 100 \
    --batch_size 2 \
    --device cuda \
    --raw_dir  data/raw/realestate \
    --proc_dir data/processed_test \
    --vae_name "Wan-AI/Wan2.2-T2V-1.3B" \
    --t5_name  "google/t5-v1_1-xxl"
```

---

## Yêu cầu cài đặt

```bash
pip install -r requirements.txt
```

> **ffmpeg** cần cài riêng và thêm vào PATH:  
> Windows: https://ffmpeg.org/download.html → ffmpeg-release-essentials.zip

---

## Yêu cầu phần cứng

| Chế độ | RAM | VRAM | Disk |
|--------|-----|------|------|
| **Dummy data** (test code) | 8 GB | ❌ không cần | < 1 GB |
| **Train thật** (100 videos) | 32 GB | 16 GB+ | ~50 GB |
| **Paper gốc** (50K videos) | 512 GB | 64× MI250 | 1 TB+ |

**Breakdown VRAM khi preprocessing:**
```
Wan2.2 VAE (fp16)   ~  3 GB
T5-XXL     (fp16)   ~ 11 GB
Overhead            ~  2 GB
────────────────────────────
Tổng                ~ 16 GB
```

**Breakdown VRAM khi training (model dim=1024):**
```
Spatia model (fp32) ~  1.5 GB
Gradient + optimizer ~  3.0 GB
Activations (bs=2)  ~  2.0 GB
────────────────────────────
Tổng                ~  6.5 GB   ← RTX 3070 8GB chạy được với bs=1
```

---

## Kiến trúc Spatia

### Tổng quan

```
Input video V  →  split thành:
  {T}^N  target clip        (cần generate)
  {P}^M  preceding clip     (context thời gian)
  {C}^O  candidate frames   (dùng để retrieve reference)

Conditioning signals:
  x_T, x_P   ← Wan2.2 VAE encode video
  x_R         ← K=7 reference frames (retrieve theo spatial overlap)
  x_S_T, x_S_P ← scene point cloud projection (qua MapAnything)
  text        ← T5 text embedding
```

### Network Block (×8)

```
1 ControlNet block ┐                    scene tokens (x_S_P, x_S_T)
                   ↓ MLP projector
4 Main blocks      → Self-Attn → Cross-Attn(text) → FFN → + scene_cond
```

### Training — 2 giai đoạn

| | Stage 1 | Stage 2 |
|-|---------|---------|
| **Freeze** | Main blocks | ControlNet blocks |
| **Train** | ControlNet blocks | Main blocks (LoRA rank=64) |
| **LR** | 1e-5 | 1e-4 |
| **Iters (paper)** | 8 000 | 5 000 |
| **Optimizer** | AdamW | AdamW |
| **LR schedule** | Cosine decay | Cosine decay |

### Flow Matching Loss

```
x_0  ~ N(0, I)                          # Gaussian noise
t    ~ logit-normal(0, 1)               # timestep sampling
x_t  = (1-t)*x_0 + t*x_T               # linear interpolation
u_t  = x_T - x_0                        # ground-truth velocity
Loss = MSE( v_θ(x_t, cond, t),  u_t )
```

### Preceding-frame Augmentation (Section 6)

Giảm khoảng cách train/inference (train dùng GT frames, inference dùng generated frames):

```
t_aug ~ Uniform[0, 0.05]
x_P_aug = (1 - t_aug)*x_P + t_aug*ε,   ε ~ N(0, I)
```

---

## Hyperparameters (paper defaults — file `configs/config.py`)

| Hyperparameter | Giá trị |
|---------------|---------|
| Model params | 5 B (paper) / ~200 M (dim=1024, code này) |
| Network blocks | 8 |
| Main blocks / network block | 4 |
| LoRA rank | 64 |
| Stage 1 LR | 1e-5 |
| Stage 2 LR | 1e-4 |
| Batch size | 64 (paper) / 2 (default code) |
| Reference frames K | 7 |
| Target frames (1st iter) | 81 |
| Preceding frames | 9 |
| Dataset | RealEstate10K (40K) + SpatialVID HD (10K) |
| Resolution | 720P |

---

## Dataset

**RealEstate10K** — tự động tải trong `run.py`:
- ~7 000 video clips trong test split
- Script tải 100 clips (theo paper dùng 100 để đánh giá)
- Source: YouTube, cần `yt-dlp` + `ffmpeg`

---

## Citation

```bibtex
@article{zhao2024spatia,
  title   = {Spatia: Video Generation with Updatable Spatial Memory},
  author  = {Zhao, Jinjing and Wei, Fangyun and Liu, Zhening and
             Zhang, Hongyang and Xu, Chang and Lu, Yan},
  journal = {arXiv preprint arXiv:2512.15716},
  year    = {2024}
}
```
