"""spatia_pipeline/setup/env_setup.py — offline shims and environment configuration."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def set_offline_env() -> None:
    """Set HuggingFace offline mode env vars."""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print("Offline mode: TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1")


def create_offline_shims(offline_pkgs_dir: Path) -> None:
    """
    Create minimal fallback Python files for packages that may not be
    installed in offline Kaggle environments:
      - loralib  (needed by ReferDINO)
      - addict
      - termcolor
      - colorlog
      - yapf
    """
    offline_pkgs_dir = Path(offline_pkgs_dir)
    offline_pkgs_dir.mkdir(parents=True, exist_ok=True)

    if str(offline_pkgs_dir) not in sys.path:
        sys.path.insert(0, str(offline_pkgs_dir))

    # 1. loralib
    (offline_pkgs_dir / "loralib.py").write_text(
        r'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1,
                 lora_dropout=0.0, fan_in_fan_out=False, merge_weights=True, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r and r > 0 else 1
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        if r and r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if getattr(self, "r", 0) and self.r > 0 and hasattr(self, "lora_A"):
            after_A = F.linear(self.lora_dropout(x), self.lora_A)
            result = result + F.linear(after_A, self.lora_B) * self.scaling
        return result

def mark_only_lora_as_trainable(model, bias="none"):
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n

def lora_state_dict(model, bias="none"):
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
''',
        encoding="utf-8",
    )

    # 2. addict
    (offline_pkgs_dir / "addict.py").write_text(
        r'''
class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, v):
        if isinstance(v, dict) and not isinstance(v, Dict):
            return cls(v)
        if isinstance(v, list):
            return [cls._wrap(x) for x in v]
        return v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)
''',
        encoding="utf-8",
    )

    # 3. termcolor
    (offline_pkgs_dir / "termcolor.py").write_text(
        'def colored(text, color=None, on_color=None, attrs=None):\n    return text\n',
        encoding="utf-8",
    )

    # 4. colorlog
    (offline_pkgs_dir / "colorlog.py").write_text(
        "import logging\nclass ColoredFormatter(logging.Formatter):\n    pass\n",
        encoding="utf-8",
    )

    # 5. yapf
    yapf_dir = offline_pkgs_dir / "yapf" / "yapflib"
    yapf_dir.mkdir(parents=True, exist_ok=True)
    (offline_pkgs_dir / "yapf" / "__init__.py").write_text("", encoding="utf-8")
    (yapf_dir / "__init__.py").write_text("", encoding="utf-8")
    (yapf_dir / "yapf_api.py").write_text(
        "def FormatCode(code, *args, **kwargs):\n    return code, False\n",
        encoding="utf-8",
    )

    print("Offline fallback packages created at:", offline_pkgs_dir)
