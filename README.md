# Spatia — Hướng dẫn chạy training pipeline

Pipeline training model **Spatia** (Wan2.2-backed spatial memory + LoRA) dựa trên paper _WorldScore_ với dữ liệu **RealEstate10K**.

---

## Cấu trúc project

```
Spatia/
├── source/
│   ├── train.py                    ← Entry point chính
│   ├── download_assets.py          ← Tải models về máy có Internet
│   ├── requirements.txt
│   └── spatia_pipeline/
│       ├── config.py               ← SpatiaConfig — toàn bộ hyperparameters
│       ├── args.py                 ← CLI arguments
│       ├── runner.py               ← Orchestrator (không dùng exec)
│       ├── setup/                  ← Kiểm tra dependencies, offline shims
│       ├── assets/                 ← Auto-detect model paths (Kaggle/local)
│       ├── data/                   ← manifest, video_utils, dataset
│       ├── preprocessing/          ← Keye-VL, ReferDINO, MapAnything
│       ├── model/                  ← LatentSpatiaControlNet, WanSpatiaTrainer
│       ├── training/               ← train_loop, checkpoint, evaluate
│       └── evaluation/             ← PSNR, SSIM, LPIPS, benchmark
└── README.md
```

---

## Phần 1 — Chạy trên máy có Internet (Local / Colab / RunPod)

### 1.1. Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -U pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r source/requirements.txt
```

### 1.2. Tải assets về máy

```bash
python source/download_assets.py --assets-dir ./assets
```

Script này tải:
- **Wan2.2** diffusers model (`wan2.2-ti2v-5b-diffusers/`)
- **ReferDINO** repo + checkpoint (`ryt_mevis_swinb.pth`)
- **MapAnything** repo + model (`Map-anything-v1/`)
- **Keye-VL** model (tuỳ chọn)

Build GroundingDINO CUDA ops (cần một lần):

```bash
cd assets/repos/ReferDINO/models/GroundingDINO/ops
python setup.py build_ext --inplace
cd -
```

### 1.3. Chuẩn bị dataset RealEstate10K

Đặt theo một trong hai cấu trúc sau:

```text
# Cách 1 — video / poses tách folder
data/realestate10k/
├── videos/
│   ├── 000001.mp4
│   └── ...
└── poses/
    ├── 000001.txt
    └── ...

# Cách 2 — cùng folder
data/realestate10k/
├── 000001.mp4
├── 000001.txt
└── ...
```

Pipeline chỉ lấy video có cả file `.mp4` lẫn `.txt` pose tương ứng. Vì nhiều link YouTube trong RealEstate10K đã bị xóa, nên chuẩn bị **dư 2× số video cần**.

### 1.4. Chạy training

```bash
python source/train.py \
  --assets-dir ./assets \
  --data-root ./data/realestate10k \
  --work-dir ./outputs \
  --train-videos 100 \
  --test-videos 20 \
  --stage both
```

Chỉ preprocess (không train):

```bash
python source/train.py \
  --assets-dir ./assets \
  --data-root ./data/realestate10k \
  --preprocess-only
```

Chạy từng stage riêng:

```bash
# Stage 1 — train control branch
python source/train.py --assets-dir ./assets --data-root ./data/realestate10k \
  --stage stage1 --stage1-steps 800

# Stage 2 — fine-tune Wan LoRA
python source/train.py --assets-dir ./assets --data-root ./data/realestate10k \
  --stage stage2 --stage2-steps 500
```

---

## Phần 2 — Chạy trên Kaggle (Offline / Internet OFF)

### 2.1. Các input cần add vào Kaggle Notebook

Trong tab **Add Input**, thêm:

| Input | Nội dung |
|---|---|
| Dataset | `realestate-small-dataset/` (video + pose) |
| Dataset | ReferDINO repo đã patch (`kaggle_bootstrap_referdino.py`) |
| Model | `wan2.2-ti2v-5b-diffusers/` (diffusers format với `model_index.json`) |
| Model | `map-anything-v1/` |
| Model | `referdino-ryt-mevis-swinb/` (`ryt_mevis_swinb.pth`) |
| Model | Keye-VL model (tuỳ chọn) |

Kaggle mount input vào `/kaggle/input/<slug>/`.

### 2.2. Chạy từ Kaggle Notebook

Pipeline tự **auto-detect** tất cả paths từ `/kaggle/input/`. Chỉ cần:

```bash
python /kaggle/working/source/train.py \
  --work-dir /kaggle/working \
  --train-videos 100 \
  --test-videos 20 \
  --offline
```

Nếu muốn chỉ định path thủ công:

```bash
python /kaggle/working/source/train.py \
  --data-root /kaggle/input/realestate-small-dataset \
  --wan-model /kaggle/input/models/wan2-ti2v-5b/pytorch/default/1 \
  --referdino-repo /kaggle/input/referdino-patched \
  --referdino-ckpt /kaggle/input/models/referdino-ryt-mevis-swinb/pytorch/default/1/ryt_mevis_swinb.pth \
  --mapanything-repo /kaggle/input/map-anything \
  --mapanything-model /kaggle/input/models/map-anything-v1/pytorch/default/1/Map-anything-v1 \
  --work-dir /kaggle/working \
  --offline
```

### 2.3. Output sau khi chạy

```text
/kaggle/working/
├── processed_spatia_full_<run_tag>/
│   └── *.pt                        ← Cached preprocessed samples
├── spatia_full_checkpoints_<run_tag>/
│   ├── stage1_step_*_wan_trainable.pt
│   ├── stage2_step_*_wan_trainable.pt
│   ├── spatia_wan2_trainable_delta_final.pt   ← Checkpoint cuối
│   ├── spatia_wan2_config.json
│   ├── benchmark_results.csv                  ← Nếu --run-benchmark
│   └── benchmark_summary.json
└── spatia_full_samples_<run_tag>/
    └── wan_spatia_reconstruction.mp4          ← Nếu --save-sample
```

### 2.4. Kiểm tra sample đã preprocess

```python
import torch

sample = torch.load("processed_spatia_full_xxx/000001.pt", map_location="cpu")
print(sample.keys())
# dict_keys(['id', 'prev', 'target', 'control', 'memory',
#            'dynamic_mask', 'reference', 'prompt', 'entities',
#            'video_path', 'pose_path', 'module_meta'])

print("target:", sample["target"].shape)   # [T, C, H, W]
print("dynamic_mask:", sample["dynamic_mask"].shape)  # [T, 1, H, W]
print("module_meta keys:", list(sample["module_meta"].keys()))
```

---

## Phần 3 — CLI Reference

```
python source/train.py [OPTIONS]

Đường dẫn:
  --data-root PATH        Dataset RealEstate10K
  --work-dir PATH         Output/checkpoint directory
  --assets-dir PATH       Folder tạo bởi download_assets.py
  --wan-model PATH        Wan2.2 diffusers directory
  --referdino-repo PATH   ReferDINO repository root
  --referdino-ckpt PATH   ryt_mevis_swinb.pth
  --mapanything-repo PATH map-anything repository root
  --mapanything-model PATH map-anything model directory
  --keye-model PATH       Keye-VL model directory

Dataset:
  --train-videos INT      Số video train (default: 100)
  --test-videos  INT      Số video validation (default: 20)
  --height INT            Frame height (default: 192)
  --width  INT            Frame width  (default: 320)
  --prev-frames      INT  (default: 9)
  --target-frames    INT  (default: 49)
  --candidate-frames INT  (default: 16)
  --ref-frames       INT  (default: 7)

Training:
  --stage {stage1,stage2,both,none}   (default: both)
  --stage1-steps INT      Stage 1 steps (default: 800)
  --stage2-steps INT      Stage 2 steps (default: 500)
  --lr-stage1 FLOAT       (default: 5e-6)
  --lr-stage2 FLOAT       (default: 1e-6)
  --batch-size INT        (default: 1)
  --grad-accum-steps INT  (default: 4)
  --lora-rank INT         (default: 64)
  --lora-alpha INT        (default: 128)
  --lora-dropout FLOAT    (default: 0.05)

Module switches:
  --run-keye BOOL         Dùng Keye-VL (default: false)
  --run-referdino BOOL    Dùng ReferDINO (default: true)
  --run-mapanything BOOL  Dùng MapAnything (default: true)

Tuỳ chọn:
  --preprocess BOOL       Chạy preprocess (default: true)
  --preprocess-only       Chỉ preprocess, không train
  --offline               Set HF_HUB_OFFLINE=1
  --run-benchmark         Chạy benchmark sau train
  --save-sample           Lưu video reconstruction mẫu
  --audit                 Kiểm tra paper-compliance
  --seed INT              (default: 42)
```

---

## Phần 4 — Gợi ý cấu hình VRAM

| VRAM | Cấu hình gợi ý |
|------|----------------|
| 16 GB | `--height 144 --width 256 --target-frames 25 --batch-size 1 --grad-accum-steps 8` |
| 24 GB | `--height 192 --width 320 --target-frames 49 --batch-size 1 --grad-accum-steps 4` *(default)* |
| 40 GB | `--height 256 --width 448 --target-frames 49 --batch-size 1 --grad-accum-steps 2` |
| 80 GB | `--height 256 --width 448 --target-frames 65 --batch-size 2 --grad-accum-steps 2` |

---

## Phần 5 — Import trực tiếp (API)

Pipeline cũng có thể được dùng như Python library:

```python
from spatia_pipeline import SpatiaConfig, WanSpatiaTrainer
from spatia_pipeline import train_loop, set_stage1_trainable, set_stage2_trainable
from spatia_pipeline import run_benchmark, compute_psnr

# Khởi tạo config
cfg = SpatiaConfig.default()

# Load model
model = WanSpatiaTrainer(cfg)

# Hoặc chạy toàn bộ pipeline
from spatia_pipeline.runner import run
state = run()  # parse từ sys.argv
```

---

## Phần 6 — Checklist trước khi chạy

- [ ] Có ít nhất `train_videos + test_videos` cặp video/pose hợp lệ
- [ ] Wan2.2 directory có `model_index.json`
- [ ] ReferDINO repo có `models/GroundingDINO/ops/setup.py`
- [ ] ReferDINO checkpoint `ryt_mevis_swinb.pth` tồn tại
- [ ] MapAnything model directory tồn tại
- [ ] GroundingDINO CUDA ops đã build (`*.so` / `*.pyd`)
- [ ] `python source/train.py --help` chạy không lỗi
- [ ] `processed_spatia_full_xxx/*.pt` được tạo sau preprocess
- [ ] Checkpoint `spatia_wan2_trainable_delta_final.pt` được lưu

### Warnings có thể bỏ qua

```
UNEXPECTED cls.predictions...
torch.meshgrid: indexing argument
torch.utils.checkpoint: use_reentrant parameter
None of the inputs have requires_grad=True
torch.cuda.amp.autocast is deprecated
```

### Errors cần sửa

```
cfg.wan_dir must be set                   → Wan2.2 model không tìm thấy
No LoRA parameters found                  → peft không cài hoặc transformer không support add_adapter
Too few processed samples                 → Kiểm tra data_root, đủ video/pose không
loss has no grad_fn                       → Validation đã để model ở eval mode, khởi động lại
```

---

## Ghi chú về dữ liệu test theo paper

Nếu cần đúng 100 video test như paper, chuẩn bị **200–250** video/link vì nhiều link RealEstate10K đã bị xóa hoặc private trên YouTube.

```text
Muốn 100 video train → chuẩn bị 200+ link ban đầu
Muốn train 100 + val 20 → chuẩn bị 250+ link ban đầu
```

Nếu chỉ còn 94 video hợp lệ, có thể ghi trong báo cáo:

> _Do một số video RealEstate10K không còn khả dụng trên YouTube, thí nghiệm sử dụng 94 video hợp lệ sau bước lọc dữ liệu._
