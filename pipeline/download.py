"""
pipeline/download.py
─────────────────────────────────────────────────────────
Step 1: Tải metadata RealEstate10K test split
Step 2: Tải video clips qua yt-dlp
"""

import random
import zipfile
import subprocess
import urllib.request
from pathlib import Path


REALESTATE_TEST_URL = (
    "https://storage.googleapis.com/realestate10k/RealEstate10K/test.zip"
)


# ── Step 1: Metadata ───────────────────────────────────────────────────────
def download_metadata(raw_dir: Path) -> Path:
    meta_dir = raw_dir / "test"
    if meta_dir.exists() and any(meta_dir.glob("*.txt")):
        n = len(list(meta_dir.glob("*.txt")))
        print(f"[Download] Metadata already exists ({n} sequences). Skipping.")
        return meta_dir

    zip_path = raw_dir / "test.zip"
    if not zip_path.exists():
        print("[Download] Fetching test metadata ...")
        urllib.request.urlretrieve(REALESTATE_TEST_URL, zip_path)

    print("[Download] Extracting metadata ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    n = len(list(meta_dir.glob("*.txt")))
    print(f"[Download] {n} test sequences ready.")
    return meta_dir


def parse_txt(txt_path: Path):
    """Returns (youtube_url, list_of_timestamps_ms)."""
    lines = txt_path.read_text().strip().splitlines()
    url   = lines[0].strip()
    ts    = [int(l.split()[0]) for l in lines[1:] if l.strip()]
    return url, ts


# ── Step 2: Videos ────────────────────────────────────────────────────────
def _download_one(url: str, video_id: str,
                  video_dir: Path, start_ms: int, end_ms: int,
                  height: int) -> Path | None:
    out = video_dir / f"{video_id}.mp4"
    if out.exists() and out.stat().st_size > 10_000:
        return out

    start_s = start_ms / 1000.0
    dur_s   = max((end_ms - start_ms) / 1000.0, 5.0)
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "-f", f"bestvideo[height<={height}][ext=mp4]/bestvideo[height<={height}]",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args",
        f"ffmpeg_i:-ss {start_s:.3f} -t {dur_s:.3f}",
        "-o", str(out), url,
    ]
    try:
        r = subprocess.run(cmd, timeout=90, capture_output=True)
        if r.returncode == 0 and out.exists():
            return out
    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found. Run: pip install yt-dlp")
    if out.exists():
        out.unlink()
    return None


def download_videos(meta_dir: Path, video_dir: Path,
                    max_videos: int, height: int) -> list[tuple[Path, Path]]:
    """
    Returns list of (video_path, txt_path) for successfully downloaded clips.
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    txts = sorted(meta_dir.glob("*.txt"))
    random.seed(42)
    random.shuffle(txts)
    txts = txts[:max_videos]

    print(f"[Download] Downloading {len(txts)} videos ...")
    results = []
    for i, tp in enumerate(txts):
        url, ts = parse_txt(tp)
        if len(ts) < 2:
            continue
        vp = _download_one(url, tp.stem, video_dir, ts[0], ts[-1], height)
        if vp:
            results.append((vp, tp))
        print(f"  {i+1:3d}/{len(txts)} | ok={len(results)}", end="\r")

    print(f"\n[Download] {len(results)} videos saved → {video_dir}")
    return results
