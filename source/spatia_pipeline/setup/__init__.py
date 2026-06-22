"""spatia_pipeline/setup/__init__.py"""
from spatia_pipeline.setup.dependencies import check_and_install
from spatia_pipeline.setup.env_setup import create_offline_shims, set_offline_env

__all__ = ["check_and_install", "create_offline_shims", "set_offline_env"]
