from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, get_args

import rasterio
import torch

from geosave_engine.geodata.datasets.base_dataset import BaseDataset, extract_key, filter_by_split

LayerName = str
RasterExtension = Literal[".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2"]
RASTER_EXTENSIONS: tuple[RasterExtension, ...] = get_args(RasterExtension)


class NonGeoDataset(BaseDataset):
    """Non-georeferenced dataset over plain raster files, one file per sample.

    One instance reads exactly one file extension — no mixed jpg/tif/png
    globbing in a single dataset. An image+label pair (different files) is
    two `NonGeoDataset` instances joined by `IntersectionDataset` on their
    shared stem key.

    Args:
        root: Directory to `rglob` for files matching `extension`.
        extension: File extension to read — one of `RASTER_EXTENSIONS`.
        layer_name: Dict key `render()` returns the tensor under.
        dtype_override: Torch dtype to cast the loaded tensor to, if it
            should differ from the loaded dtype.
        key_pattern: Regex to extract the sample key from each file's name.
            None strips the extension and uses the rest — see `extract_key`.
        split: Text file of file stems to keep, one per line. None keeps
            every matching file found under `root`.

    Raises:
        ValueError: `extension` isn't one of `RASTER_EXTENSIONS`.
    """

    def __init__(
        self,
        root: str | Path,
        extension: RasterExtension = ".tif",
        *,
        layer_name: LayerName = "image",
        dtype_override: torch.dtype | None = None,
        key_pattern: str | None = None,
        split: str | Path | None = None,
    ) -> None:
        if extension not in RASTER_EXTENSIONS:
            raise ValueError(f"extension must be one of {RASTER_EXTENSIONS}, got {extension!r}")

        self.split = split
        self.root = Path(root)
        self.extension = extension
        self.layer_name = layer_name
        self.dtype_override = dtype_override
        self.key_pattern = key_pattern

        paths: dict[str, Path] = {
            extract_key(p.name, key_pattern): p for p in sorted(self.root.rglob(f"*{extension}"))
        }
        self.paths = filter_by_split(paths, split)
        self.reindex(self.paths)

    def render(self, key: str) -> dict[str, Any]:
        """Render one sample.

        Args:
            key: File stem — one of `self.keys`.

        Returns:
            `{layer_name: Tensor}`, shape `[C, H, W]`.
        """
        with rasterio.open(self.paths[key]) as src:
            array = src.read()
        tensor = torch.from_numpy(array)
        if self.dtype_override is not None:
            tensor = tensor.to(self.dtype_override)
        return {self.layer_name: tensor}

    def to_row(self, key: str) -> dict[str, Any]:
        """Manifest row for `key`.

        Args:
            key: File stem — one of `self.keys`.

        Returns:
            `{"path": ...}` — `self.paths[key]` relative to `self.root`.
        """
        return {"path": str(self.paths[key].relative_to(self.root))}
