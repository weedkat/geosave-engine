from __future__ import annotations

from pathlib import Path

import torch


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "geosave" / "pretrained"


def download_weights(url: str, destination: Path) -> Path:
    """Download weights from `url` to `destination`, skipping if already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    torch.hub.download_url_to_file(url, str(destination))
    return destination


def cached_weights_path(cache_dir: Path, name: str, *, suffix: str = ".pth") -> Path:
    """Return the canonical cache path for weights named `name`."""
    return cache_dir / f"{name}{suffix}"
