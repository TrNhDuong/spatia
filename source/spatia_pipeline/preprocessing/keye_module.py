"""
spatia_pipeline/preprocessing/keye_module.py
----------------------------------------------
Lazy-loading wrapper for Keye-VL scene description model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class KeyeModule:
    """
    Lazy wrapper around the Keye-VL-1.5-8B vision-language model.

    Usage::

        keye = KeyeModule(model_path="/path/to/keye", device=device, enabled=True)
        result = keye.describe_video("clip.mp4", fallback_prompt="A real estate video.")
    """

    DEFAULT_PROMPT = "A realistic real estate video with smooth camera movement."

    def __init__(
        self,
        model_path: Optional[Path | str],
        device: torch.device,
        enabled: bool = False,
        strict: bool = False,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.device     = device
        self.enabled    = enabled
        self.strict     = strict

        self._model     = None
        self._processor = None
        self._loaded    = False
        self._error: Optional[str] = None

    # ------------------------------------------------------------------

    def _load(self) -> Tuple[Optional[object], Optional[object]]:
        if self._loaded:
            return self._model, self._processor

        self._loaded = True
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore

            model_id   = str(self.model_path) if self.model_path is not None else "Kwai-Keye/Keye-VL-1_5-8B"
            local_only = self.model_path is not None
            print("Loading Keye-VL from", model_id)

            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype="auto",
                local_files_only=local_only,
                device_map="auto" if self.device.type == "cuda" else None,
            ).eval()

            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=local_only
            )
            self._model     = model
            self._processor = processor
            return model, processor

        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            print("WARN Keye load failed:", self._error)
            if self.strict:
                raise
            return None, None

    # ------------------------------------------------------------------

    def describe_video(
        self,
        video_path: str | Path,
        fallback_prompt: str,
    ) -> Dict:
        """
        Run Keye-VL to get a scene description + entity list.

        Returns a dict with keys: prompt, entities, source.
        Falls back gracefully if disabled or model unavailable.
        """
        if not self.enabled:
            return {
                "prompt":   self.DEFAULT_PROMPT,
                "entities": ["moving objects"],
                "source":   "fallback_disabled",
            }

        model, processor = self._load()
        if model is None or processor is None:
            return {
                "prompt":   fallback_prompt,
                "entities": ["moving objects"],
                "source":   "fallback_load_failed",
                "error":    self._error,
            }

        try:
            from keye_vl_utils import process_vision_info  # type: ignore

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {
                        "type": "text",
                        "text": (
                            "Describe this real-estate video in one concise generation prompt, "
                            "then list dynamic entities as comma-separated nouns. /no_think"
                        ),
                    },
                ],
            }]
            text            = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_in, vid_in = process_vision_info(messages)
            inputs          = processor(
                text=[text], images=image_in, videos=vid_in,
                padding=True, return_tensors="pt"
            )
            inputs = inputs.to(next(model.parameters()).device)

            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)

            out = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            # Simple entity parser
            entities: List[str] = []
            for part in re.split(r"[,;\n]", out):
                s = part.strip().lower()
                if 1 <= len(s.split()) <= 5 and any(
                    w in s
                    for w in ["person", "car", "animal", "object", "door",
                               "curtain", "tree", "moving"]
                ):
                    entities.append(s)
            entities = entities[:5] or ["moving objects"]

            return {
                "prompt":   out[:400] if out else fallback_prompt,
                "entities": entities,
                "source":   "keye",
            }

        except Exception as e:
            print("WARN Keye inference failed:", type(e).__name__, e)
            if self.strict:
                raise
            return {
                "prompt":   fallback_prompt,
                "entities": ["moving objects"],
                "source":   "fallback_infer_failed",
                "error":    f"{type(e).__name__}: {e}",
            }

    def unload(self) -> None:
        """Release model memory."""
        self._model     = None
        self._processor = None
        self._loaded    = False
