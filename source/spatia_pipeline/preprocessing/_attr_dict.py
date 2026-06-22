"""
spatia_pipeline/preprocessing/_attr_dict.py
--------------------------------------------
Minimal EasyDict-like class used as a fallback when easydict is not installed.
"""
from __future__ import annotations


class AttrDict(dict):
    """Recursive attribute-access dict."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = self._wrap(v)

    @staticmethod
    def _wrap(v):
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            return AttrDict(v)
        if isinstance(v, list):
            return [AttrDict._wrap(x) for x in v]
        return v

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = self._wrap(value)
