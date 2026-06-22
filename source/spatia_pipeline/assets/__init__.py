"""spatia_pipeline/assets/__init__.py"""
from spatia_pipeline.assets.path_finder import (
    autodetect_local_paths,
    find_keye_model,
    find_mapanything_model,
    find_mapanything_repo,
    find_patched_referdino_repo,
    find_referdino_ckpt,
    find_wan_model,
    resolve_diffusers_model_dir,
    add_repo_paths,
)

__all__ = [
    "autodetect_local_paths",
    "find_keye_model",
    "find_mapanything_model",
    "find_mapanything_repo",
    "find_patched_referdino_repo",
    "find_referdino_ckpt",
    "find_wan_model",
    "resolve_diffusers_model_dir",
    "add_repo_paths",
]
