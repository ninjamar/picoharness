from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Configuration

_config: Configuration | None = None


def set_config(config: Configuration) -> None:
    global _config
    _config = config


def get_config() -> Configuration:
    if _config is None:
        raise RuntimeError("Configuration not initialized — call set_config() first")
    return _config
