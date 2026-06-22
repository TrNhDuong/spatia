"""spatia_pipeline/setup/dependencies.py — package dependency checker."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Dict, List


REQUIRED_PACKAGES: List[str] = [
    "tqdm",
    "opencv-python",
    "pandas",
    "numpy",
    "Pillow",
    "imageio",
    "safetensors",
]

OPTIONAL_PACKAGES: List[str] = [
    "transformers",
    "accelerate",
    "keye-vl-utils",
    "ruamel.yaml",
    "easydict",
]

_PKG_MODULE_MAP: Dict[str, str] = {
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "ruamel.yaml": "ruamel",
    "keye-vl-utils": "keye_vl_utils",
}


def module_name(pkg: str) -> str:
    return _PKG_MODULE_MAP.get(pkg, pkg.replace("-", "_"))


def pip_install(pkg: str) -> None:
    print("Installing", pkg)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-input", "-q", pkg,
    ])


def check_and_install(
    required: List[str] = REQUIRED_PACKAGES,
    optional: List[str] = OPTIONAL_PACKAGES,
    auto_install_required: bool = True,
) -> None:
    """Check all required packages (installing if missing) and report optional ones."""
    print("Checking required packages...")
    for pkg in required:
        mod = module_name(pkg)
        if importlib.util.find_spec(mod) is None:
            if auto_install_required:
                try:
                    pip_install(pkg)
                except Exception as e:
                    print(f"WARN required install failed: {pkg} {type(e).__name__}: {e}")
            else:
                print(f"MISSING  {pkg}")
        else:
            print("OK", pkg)

    print("\nChecking optional packages (no auto-install)...")
    for pkg in optional:
        mod = module_name(pkg)
        ok = importlib.util.find_spec(mod) is not None
        print(("OK       " if ok else "MISSING  "), pkg)

    print("\nDone. Missing optional packages are allowed.")
