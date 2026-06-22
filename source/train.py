"""
source/train.py
----------------
Entry point for the Spatia training pipeline.

Usage::

    python train.py --assets-dir /workspace/assets --data-root /data/realestate10k
    python train.py --stage stage1 --stage1-steps 1000
    python train.py --preprocess-only
    python train.py --help
"""

import sys
from pathlib import Path

# Ensure `source/` is importable as a package root
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from spatia_pipeline.args import parse_args
from spatia_pipeline.runner import run


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
