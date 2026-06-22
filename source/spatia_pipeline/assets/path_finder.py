"""
spatia_pipeline/assets/path_finder.py
--------------------------------------
Auto-detect model/repo/dataset paths from Kaggle inputs or local directories.
Consolidates logic previously spread across notebook cells 2, 5, 6, 8, 9.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = (
    Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/workspace")
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def list_input_roots(root: Path = KAGGLE_INPUT) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def has_any_file(root: Optional[Path], patterns: List[str]) -> bool:
    if root is None:
        return False
    root = Path(root)
    if root.is_file():
        return any(root.match(pat) for pat in patterns)
    if not patterns:
        return True
    return any(bool(list(root.rglob(pat))) for pat in patterns)


def find_dir_by_keywords(
    keywords: List[str],
    must_have_any: Optional[List[str]] = None,
    root: Path = KAGGLE_INPUT,
) -> Optional[Path]:
    if not root.exists():
        return None
    keywords_l = [k.lower() for k in keywords]
    candidates: List[Path] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        text = str(d).lower()
        if all(k in text for k in keywords_l):
            if must_have_any is None or has_any_file(d, must_have_any):
                candidates.append(d)
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0] if candidates else None


def find_file_by_keywords(
    keywords: List[str],
    patterns: List[str],
    root: Path = KAGGLE_INPUT,
) -> Optional[Path]:
    if not root.exists():
        return None
    keywords_l = [k.lower() for k in keywords]
    candidates: List[Path] = []
    for pat in patterns:
        for f in root.rglob(pat):
            if all(k in str(f).lower() for k in keywords_l):
                candidates.append(f)
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# ReferDINO
# ---------------------------------------------------------------------------

def _is_referdino_repo(p: Optional[Path]) -> bool:
    if p is None:
        return False
    p = Path(p)
    return (
        (p / "models" / "GroundingDINO" / "ops" / "setup.py").exists()
        and (p / "models").exists()
    )


def _is_patched_referdino_repo(p: Optional[Path]) -> bool:
    if p is None:
        return False
    return _is_referdino_repo(p) and (Path(p) / "kaggle_bootstrap_referdino.py").exists()


def find_patched_referdino_repo(root: Path = KAGGLE_INPUT) -> Optional[Path]:
    """
    Locate the patched ReferDINO repo (contains kaggle_bootstrap_referdino.py).
    Falls back to extracting a zip if the directory is not already extracted.
    """
    if not root.exists():
        return None

    # 1. Direct hit
    direct: List[Path] = []
    for f in root.rglob("kaggle_bootstrap_referdino.py"):
        repo = f.parent
        if _is_patched_referdino_repo(repo):
            direct.append(repo)
    if direct:
        return sorted(direct, key=lambda p: (len(p.parts), str(p)))[0]

    # 2. Extract from zip
    for z in sorted(root.rglob("*.zip"), key=lambda p: (len(p.parts), str(p))):
        if not any(k in str(z).lower() for k in ["refer", "dino"]):
            continue
        extract_root = KAGGLE_WORKING / "extracted_inputs" / z.stem
        marker = extract_root / ".extracted_ok"
        try:
            if not marker.exists():
                if extract_root.exists():
                    shutil.rmtree(extract_root)
                extract_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z, "r") as zipf:
                    zipf.extractall(extract_root)
                marker.write_text("ok", encoding="utf-8")
            for f in extract_root.rglob("kaggle_bootstrap_referdino.py"):
                repo = f.parent
                if _is_patched_referdino_repo(repo):
                    print("Extracted patched ReferDINO zip:", z)
                    return repo
        except Exception as e:
            print(f"WARN cannot extract ReferDINO zip: {z} {type(e).__name__}: {e}")

    # 3. Unpatched fallback
    for p in root.rglob("*"):
        if p.is_dir() and _is_referdino_repo(p):
            return p

    return None


def find_referdino_ckpt(root: Path = KAGGLE_INPUT) -> Optional[Path]:
    if not root.exists():
        return None
    for pat in ["ryt_mevis_swinb.pth", "*.pth", "*.pt", "*.ckpt"]:
        for p in root.rglob(pat):
            txt = str(p).lower()
            if any(k in txt for k in ["referdino", "mevis", "swin"]):
                return p
    return None


# ---------------------------------------------------------------------------
# MapAnything
# ---------------------------------------------------------------------------

def find_mapanything_repo(root: Path = KAGGLE_INPUT) -> Optional[Path]:
    for keywords in [["map", "anything"], ["mapanything"]]:
        hit = find_dir_by_keywords(keywords, must_have_any=["*.py"], root=root)
        if hit is not None and (hit / "mapanything").exists():
            return hit
    return None


def find_mapanything_model(root: Path = KAGGLE_INPUT) -> Optional[Path]:
    if not root.exists():
        return None
    for name in ["Map-anything-v1", "map-anything-v1", "map_anything_v1"]:
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# Wan2.2
# ---------------------------------------------------------------------------

def find_wan_model(root: Path = KAGGLE_INPUT) -> Optional[Path]:
    return find_dir_by_keywords(["wan"], ["*.safetensors", "*.json", "*.pth"], root=root)


def resolve_diffusers_model_dir(root: Optional[Path], name: str = "model") -> Path:
    """Return the directory containing model_index.json."""
    if root is None or not Path(root).exists():
        raise FileNotFoundError(f"{name} root not found: {root}")
    root = Path(root)
    if (root / "model_index.json").exists():
        return root
    hits = sorted(root.rglob("model_index.json"), key=lambda p: (len(p.parts), str(p)))
    if not hits:
        raise FileNotFoundError(f"No model_index.json found under {root}")
    return hits[0].parent


# ---------------------------------------------------------------------------
# Keye-VL
# ---------------------------------------------------------------------------

def find_keye_model(root: Path = KAGGLE_INPUT) -> Optional[Path]:
    return find_dir_by_keywords(["keye"], ["*.safetensors", "*.json", "*.py"], root=root)


# ---------------------------------------------------------------------------
# Local (non-Kaggle) detection
# ---------------------------------------------------------------------------

def autodetect_local_paths(
    assets_dir: Optional[Path] = None,
) -> dict:
    """
    When running locally (not on Kaggle), look under ``assets_dir`` for
    models/repos laid out by ``download_assets.py``.

    Returns a dict with keys: wan_model, referdino_repo, referdino_ckpt,
    mapanything_repo, mapanything_model, keye_model (each Path or None).
    """
    result: dict = {k: None for k in [
        "wan_model", "referdino_repo", "referdino_ckpt",
        "mapanything_repo", "mapanything_model", "keye_model",
    ]}

    if assets_dir is None:
        return result

    base = Path(assets_dir)

    candidates = {
        "wan_model": base / "models" / "wan2.2-ti2v-5b-diffusers",
        "referdino_repo": base / "repos" / "ReferDINO",
        "referdino_ckpt": base / "checkpoints" / "referdino" / "ryt_mevis_swinb.pth",
        "mapanything_repo": base / "repos" / "map-anything",
        "mapanything_model": base / "models" / "map-anything",
        "keye_model": base / "models" / "keye-vl-1.5-8b",
    }
    for key, path in candidates.items():
        if path.exists():
            result[key] = path

    return result


# ---------------------------------------------------------------------------
# Add repo paths to sys.path
# ---------------------------------------------------------------------------

def add_repo_paths(
    referdino_repo: Optional[Path] = None,
    mapanything_repo: Optional[Path] = None,
) -> None:
    """Insert ReferDINO / MapAnything directories into sys.path."""
    paths_to_add: List[Path] = []

    if mapanything_repo is not None:
        repo = Path(mapanything_repo)
        paths_to_add += [repo, repo.parent]

    if referdino_repo is not None:
        repo = Path(referdino_repo)
        paths_to_add += [
            repo,
            repo / "models" / "GroundingDINO",
            repo / "models" / "GroundingDINO" / "ops",
        ]

    for p in paths_to_add:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            print("Added to sys.path:", p)
