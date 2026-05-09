import sys
import random
from pathlib import Path

# Add current dir to python path to import pipeline
sys.path.append(str(Path(__file__).parent))

from pipeline.download import _download_one, parse_txt
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def main():
    meta_dir = Path("RealEstate10K/RealEstate10K/test")
    video_dir = Path("data/raw/realestate/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    txts = sorted(meta_dir.glob("*.txt"))
    random.seed(42)
    random.shuffle(txts)
    
    TARGET_COUNT = 100
    print(f"Downloading {TARGET_COUNT} clips from {meta_dir} to {video_dir}...")
    
    results = []
    failed = []
    
    for tp in txts:
        if len(results) >= TARGET_COUNT:
            break
            
        url, ts_sec = parse_txt(tp)
        if len(ts_sec) < 2:
            continue
            
        vp = _download_one(
            url, tp.stem, video_dir,
            start_sec=ts_sec[0],
            end_sec=ts_sec[-1],
            height=480,
            max_retries=3,
        )
        
        if vp:
            results.append((vp, tp))
            status = "[OK]"
        else:
            failed.append(f"{tp.name}\tdownload failed")
            status = "[FAIL]"
            
        pct = len(results) / TARGET_COUNT * 100
        print(
            f"  [{len(results):4d}/{TARGET_COUNT}] {pct:5.1f}% | "
            f"ok={len(results)} fail={len(failed)} | {status} {tp.stem[:20]}",
            end="\r",
            flush=True,
        )
        
        import time
        time.sleep(3) # Tránh bị YouTube block (HTTP 429)
            
    print("\nDownload complete.")
    if failed:
        failed_log = video_dir / "failed_downloads.txt"
        failed_log.write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} videos failed → {failed_log}")

if __name__ == "__main__":
    main()
