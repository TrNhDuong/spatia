from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


def run_cmd(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), cwd=str(cwd) if cwd else None)


def ensure_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        run_cmd([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])


def hf_snapshot(repo_id: str, local_dir: Path, token: str | None = None, repo_type: str = "model", allow_patterns=None):
    ensure_hf_hub()
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HF {repo_type}: {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
    )


def git_clone(url: str, dst: Path, depth: int | None = 1):
    if dst.exists() and any(dst.iterdir()):
        print("Repo exists, skip:", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if depth:
        cmd += ["--depth", str(depth)]
    cmd += [url, str(dst)]
    run_cmd(cmd)


def download_url(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print("File exists, skip:", dst)
        return
    print(f"Downloading URL: {url} -> {dst}")
    urlretrieve(url, dst)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--assets-dir", type=str, default="assets")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))

    p.add_argument("--wan-repo-id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--mapanything-repo-id", type=str, default="facebook/map-anything")
    p.add_argument("--referdino-repo-id", type=str, default="liangtm/referdino")
    p.add_argument("--bert-repo-id", type=str, default="bert-base-uncased")
    p.add_argument("--keye-repo-id", type=str, default="", help="Optional Keye-VL HF repo id if you use --run-keye true.")

    p.add_argument("--referdino-git", type=str, default="https://github.com/iSEE-Laboratory/ReferDINO.git")
    p.add_argument("--mapanything-git", type=str, default="https://github.com/facebookresearch/map-anything.git")
    p.add_argument("--skip-wan", action="store_true")
    p.add_argument("--skip-mapanything", action="store_true")
    p.add_argument("--skip-referdino", action="store_true")
    p.add_argument("--skip-bert", action="store_true")
    args = p.parse_args()

    root = Path(args.assets_dir).expanduser().resolve()
    models = root / "models"
    repos = root / "repos"
    ckpts = root / "checkpoints"
    for d in [models, repos, ckpts]:
        d.mkdir(parents=True, exist_ok=True)

    if not args.skip_wan:
        hf_snapshot(args.wan_repo_id, models / "wan2.2-ti2v-5b-diffusers", token=args.hf_token)

    if not args.skip_mapanything:
        git_clone(args.mapanything_git, repos / "map-anything")
        hf_snapshot(args.mapanything_repo_id, models / "map-anything", token=args.hf_token)

    if not args.skip_referdino:
        git_clone(args.referdino_git, repos / "ReferDINO")
        # The notebook expects a full ReferDINO checkpoint. The official model zoo includes ryt_mevis_swinb.pth.
        hf_snapshot(
            args.referdino_repo_id,
            ckpts / "referdino",
            token=args.hf_token,
            allow_patterns=["ryt_mevis_swinb.pth", "*.md"],
        )
        # Also keep GroundingDINO pretrained weight in a separate folder for repo scripts that request it.
        download_url(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
            ckpts / "groundingdino" / "groundingdino_swinb_cogcoor.pth",
        )

    if not args.skip_bert:
        hf_snapshot(args.bert_repo_id, models / "bert-base-uncased", token=args.hf_token)

    if args.keye_repo_id:
        hf_snapshot(args.keye_repo_id, models / "keye-vl-1.5-8b", token=args.hf_token)

    manifest = root / "ASSETS_LAYOUT.txt"
    manifest.write_text(
        (
            f"Assets root: {root}\n\n"
            "Use train.py like:\n"
            f"python train.py --assets-dir {root} --data-root /path/to/realestate10k --work-dir ./outputs\n\n"
            "Important paths:\n"
            f"Wan model:          {models / 'wan2.2-ti2v-5b-diffusers'}\n"
            f"ReferDINO repo:     {repos / 'ReferDINO'}\n"
            f"ReferDINO ckpt:     {ckpts / 'referdino' / 'ryt_mevis_swinb.pth'}\n"
            f"MapAnything repo:   {repos / 'map-anything'}\n"
            f"MapAnything model:  {models / 'map-anything'}\n"
            f"BERT model:         {models / 'bert-base-uncased'}\n"
        ),
        encoding="utf-8",
    )
    print("Done. Wrote", manifest)


if __name__ == "__main__":
    main()
