# Spatia Wan2.2 Training Project

Project này được tách từ notebook `spatia-training (8).ipynb` thành dạng chạy bằng file `.py` và tham số `args`.

## Cấu trúc chính

```text
spatia_project/
├── train.py                         # chạy training bằng argparse
├── download_assets.py               # tải model/repo về folder riêng
├── requirements.txt
├── original_notebook.ipynb          # notebook gốc giữ lại để đối chiếu
├── original_notebook_export.py      # export toàn bộ code từ notebook
├── assets/
│   ├── models/
│   ├── repos/
│   └── checkpoints/
└── spatia_pipeline/
    ├── args.py
    ├── runner.py
    └── stages/                     # từng cell code đã được tách thành file riêng
```

## 1. Tải repo/model

Chạy online trước để kéo asset về một thư mục riêng:

```bash
python download_assets.py --assets-dir ./assets
```

Mặc định script tải:

- Wan2.2 Diffusers: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- MapAnything repo: `https://github.com/facebookresearch/map-anything.git`
- MapAnything model: `facebook/map-anything`
- ReferDINO repo: `https://github.com/iSEE-Laboratory/ReferDINO.git`
- ReferDINO checkpoint: `liangtm/referdino`, file `ryt_mevis_swinb.pth`
- BERT local: `bert-base-uncased`

Nếu muốn tải Keye-VL thì truyền thêm repo id bạn đang dùng:

```bash
python download_assets.py --assets-dir ./assets --keye-repo-id <HF_REPO_ID_CUA_KEYE>
```

## 2. Chạy training

Ví dụ chạy đầy đủ stage 1 + stage 2:

```bash
python train.py \
  --assets-dir ./assets \
  --data-root /path/to/realestate10k \
  --work-dir ./outputs \
  --train-videos 100 \
  --test-videos 20 \
  --height 192 \
  --width 320 \
  --prev-frames 9 \
  --target-frames 49 \
  --candidate-frames 16 \
  --ref-frames 7 \
  --stage both \
  --stage1-steps 800 \
  --stage2-steps 500
```

Chỉ tiền xử lý:

```bash
python train.py --assets-dir ./assets --data-root /path/to/realestate10k --work-dir ./outputs --preprocess-only
```

Chỉ train Stage 2:

```bash
python train.py --assets-dir ./assets --data-root /path/to/realestate10k --work-dir ./outputs --stage stage2
```

## 3. Tham số quan trọng

| Tham số | Ý nghĩa |
|---|---|
| `--data-root` | Folder dataset chứa `videos/`, `poses/`, hoặc `manifest.csv`. |
| `--assets-dir` | Folder do `download_assets.py` tạo ra. |
| `--wan-model` | Ghi đè đường dẫn Wan2.2 Diffusers nếu không dùng `--assets-dir`. |
| `--referdino-repo` / `--referdino-ckpt` | Ghi đè repo/checkpoint ReferDINO. |
| `--mapanything-repo` / `--mapanything-model` | Ghi đè repo/model MapAnything. |
| `--stage` | `stage1`, `stage2`, `both`, hoặc `none`. |
| `--run-keye` | Mặc định `False`, vì notebook gốc cũng tắt Keye. |
| `--run-referdino` | Mặc định `True`. |
| `--run-mapanything` | Mặc định `True`. |
| `--strict-external-models` | Mặc định `True`; lỗi external model sẽ dừng pipeline. |

## 4. Ghi chú Kaggle

- Nếu đã add asset qua Kaggle Dataset/Model và muốn dùng auto-detect `/kaggle/input`, có thể bỏ `--assets-dir` và truyền path thủ công bằng `--wan-model`, `--referdino-repo`, ...
- Các cell fix hardcode từ notebook cũ vẫn nằm trong `spatia_pipeline/stages/06_*` đến `23_*`, nhưng không chạy mặc định. Muốn chạy chúng thì thêm:

```bash
python train.py ... --run-legacy-kaggle-fixes
```

## 5. Output

Checkpoint cuối nằm trong:

```text
<work-dir>/spatia_full_checkpoints_<run_tag>/spatia_wan2_trainable_delta_final.pt
```

Nếu thêm `--package-output`, script sẽ đóng gói checkpoint/samples/manifest giống notebook gốc.
