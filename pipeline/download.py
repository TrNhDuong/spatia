"""
pipeline/download.py
─────────────────────────────────────────────────────────
Step 1: Metadata RealEstate10K test split (đã có sẵn nếu folder tồn tại)
Step 2: Tải video clips qua yt-dlp (parallel, retry, resume)

Fix log:
  - Bug: timestamp đơn vị microseconds → phải chia 1_000_000 để ra giây
  - Thêm parallel download (ThreadPoolExecutor)
  - Thêm retry logic (3 lần) với backoff
  - Thêm failed_downloads.txt để log video thất bại
  - Cải thiện yt-dlp format string để tương thích cao hơn
  - Hiển thị progress bar đẹp hơn
"""

import random
import zipfile
import subprocess
import time
import urllib.request
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REALESTATE_TEST_URL = (
    "https://storage.googleapis.com/realestate10k/RealEstate10K/test.zip"
)

# ── Step 1: Metadata ───────────────────────────────────────────────────────
def download_metadata(raw_dir: Path) -> Path:
    """Tải và giải nén metadata test split. Bỏ qua nếu đã tồn tại."""
    meta_dir = raw_dir / "test"
    if meta_dir.exists() and any(meta_dir.glob("*.txt")):
        n = len(list(meta_dir.glob("*.txt")))
        log.info(f"Metadata already exists ({n} sequences). Skipping.")
        return meta_dir

    zip_path = raw_dir / "test.zip"
    if not zip_path.exists():
        log.info("Fetching test metadata ...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(REALESTATE_TEST_URL, zip_path)

    log.info("Extracting metadata ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    n = len(list(meta_dir.glob("*.txt")))
    log.info(f"{n} test sequences ready.")
    return meta_dir


def parse_txt(txt_path: Path):
    """
    Trả về (youtube_url, list_of_timestamps_seconds).

    NOTE: timestamps trong .txt là MICROSECONDS (µs) tính từ đầu video,
    cần chia 1_000_000 để ra giây.
    """
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    url = lines[0].strip()
    # timestamp cột đầu là microseconds
    ts_us = [int(line.split()[0]) for line in lines[1:] if line.strip()]
    ts_sec = [t / 1_000_000.0 for t in ts_us]
    return url, ts_sec


# ── Step 2: Download một video ────────────────────────────────────────────
# Minimum clip duration đủ để preprocess:
# target_frames=81 + preceding_frames=9 + max_ref_frames*2=14 = 104 frames
# @ 24fps (paper) ≈ 4.3s → dùng 5s để có buffer an toàn
MIN_CLIP_SEC = 5.0


def _download_one(
    url: str,
    video_id: str,
    video_dir: Path,
    start_sec: float,
    end_sec: float,
    height: int,
    max_retries: int = 3,
) -> Path | None:
    """
    Tải đoạn clip [start_sec, end_sec] từ YouTube bằng yt-dlp + ffmpeg.

    Trả về Path nếu thành công, None nếu thất bại.
    """
    out = video_dir / f"{video_id}.mp4"

    # Resume: bỏ qua nếu file đã có và > 10 KB
    if out.exists() and out.stat().st_size > 10_000:
        return out

    # Đảm bảo clip đủ dài để preprocess lấy đủ frames
    duration_sec = max(end_sec - start_sec, MIN_CLIP_SEC)

    # yt-dlp format: ưu tiên mp4, fallback sang bất kỳ
    fmt = (
        f"bestvideo[height<={height}][ext=mp4]"
        f"+bestaudio[ext=m4a]"
        f"/bestvideo[height<={height}]"
        f"/best[height<={height}]"
        f"/best"
    )

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args",
        f"ffmpeg_i:-ss {start_sec:.3f} -t {duration_sec:.3f}",
        "--postprocessor-args",
        "ffmpeg:-c:v libx264 -c:a aac",  # đảm bảo codec tương thích
        "-o", str(out),
        url,
    ]

    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                timeout=120,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and out.exists() and out.stat().st_size > 10_000:
                return out
            # yt-dlp lỗi nhưng không exception
            err_msg = result.stderr.strip().splitlines()
            short_err = err_msg[-1] if err_msg else "unknown error"
            log.debug(f"[{video_id}] attempt {attempt} failed: {short_err}")

        except subprocess.TimeoutExpired:
            log.debug(f"[{video_id}] attempt {attempt} timed out")
        except FileNotFoundError:
            raise RuntimeError(
                "yt-dlp không tìm thấy. Cài đặt bằng: pip install yt-dlp"
            )

        # Xóa file lỗi nếu tồn tại
        if out.exists():
            out.unlink(missing_ok=True)

        if attempt < max_retries:
            time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s

    return None


# ── Step 2: Download nhiều video (parallel) ───────────────────────────────
def download_videos(
    meta_dir: Path,
    video_dir: Path,
    max_videos: int,
    height: int,
    workers: int = 4,
    max_retries: int = 3,
) -> list[tuple[Path, Path]]:
    """
    Tải tối đa `max_videos` clips song song với `workers` threads.

    Trả về list[(video_path, txt_path)] cho các clip tải thành công.
    Log các video thất bại vào video_dir/failed_downloads.txt.

    Args:
        meta_dir:   thư mục chứa .txt metadata
        video_dir:  thư mục lưu .mp4
        max_videos: số clip tối đa cần tải
        height:     độ phân giải tối đa (px)
        workers:    số luồng tải song song
        max_retries: số lần retry mỗi video
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    failed_log = video_dir / "failed_downloads.txt"

    # Chọn ngẫu nhiên max_videos file txt
    txts = sorted(meta_dir.glob("*.txt"))
    random.seed(42)
    random.shuffle(txts)
    txts = txts[:max_videos]

    log.info(f"Downloading {len(txts)} clips | workers={workers} | max_height={height}p")

    results: list[tuple[Path, Path]] = []
    failed: list[str] = []
    lock = Lock()
    done_count = 0

    def _task(tp: Path):
        nonlocal done_count
        url, ts_sec = parse_txt(tp)
        if len(ts_sec) < 2:
            return None, tp, "< 2 timestamps"

        vp = _download_one(
            url, tp.stem, video_dir,
            start_sec=ts_sec[0],
            end_sec=ts_sec[-1],
            height=height,
            max_retries=max_retries,
        )
        return vp, tp, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, tp): tp for tp in txts}

        for future in as_completed(futures):
            vp, tp, err = future.result()
            done_count += 1

            with lock:
                if vp:
                    results.append((vp, tp))
                    status = "✓"
                else:
                    reason = err or "download failed"
                    failed.append(f"{tp.name}\t{reason}")
                    status = "✗"

            # Progress line
            pct = done_count / len(txts) * 100
            print(
                f"  [{done_count:4d}/{len(txts)}] {pct:5.1f}% | "
                f"ok={len(results)} fail={len(failed)} | {status} {tp.stem[:20]}",
                end="\r",
                flush=True,
            )

    print()  # xuống dòng sau progress

    # Ghi log thất bại
    if failed:
        failed_log.write_text("\n".join(failed), encoding="utf-8")
        log.warning(f"{len(failed)} videos failed → {failed_log}")
    else:
        log.info("All clips downloaded successfully!")

    log.info(f"{len(results)} clips saved → {video_dir}")
    return results
