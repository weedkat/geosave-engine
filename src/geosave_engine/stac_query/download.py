from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AssetDownloader(Protocol):
    """Downloads a single asset from a STAC item to a local directory."""

    def download(self, item: Any, asset_key: str, out_dir: Path) -> Path: ...
