from __future__ import annotations

import urllib.request
from pathlib import Path

from tqdm import tqdm


def cached_weights_path(cache_dir: Path, name: str) -> Path:
    """Return path for cached weights file, creating the directory if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.pth"


def download_weights(url: str, destination: Path) -> Path:
    """Download weights from *url* to *destination*, skipping if already cached."""
    if destination.exists():
        return destination

    filename = destination.name

    class _Progress(tqdm):
        def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> bool | None:
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

    with _Progress(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, destination, reporthook=t.update_to)

    return destination
