from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, get_args

import torch

from geosave_engine.geodata.datasets.base_dataset import BaseDataset, extract_key, filter_by_split
from geosave_engine.geodata.tile import GeoTile

LayerName = str
GeoRasterExtension = Literal[".tif", ".tiff", ".zarr"]
GEO_RASTER_EXTENSIONS: tuple[GeoRasterExtension, ...] = get_args(GeoRasterExtension)


class GeoDataset(BaseDataset):
    """Georeferenced dataset over one GeoTIFF/Zarr file per sample.

    Same one-file-per-sample, one-modality-per-instance shape as
    `NonGeoDataset`, but keeps each sample's real `GeoAnchor` (CRS,
    transform, bounds) via `GeoTile.from_geotiff`/`from_zarr` instead of
    reading a bare tensor. Join several layers with `IntersectionDataset`/
    `&`, same as `NonGeoDataset`.

    A `.tif`/`.zarr` without usable CRS/geobox metadata can't go through
    `GeoTile` at all — that belongs in `NonGeoDataset` instead, read as a
    literal array.

    Args:
        root: Directory to `rglob` for files matching `extension`.
        extension: File extension to read — one of `GEO_RASTER_EXTENSIONS`.
            `.zarr` must be a store written by `GeoTile.to_zarr`.
        layer_name: Dict key `render()` returns the tensor under.
        sel_bands: Band names to keep; None keeps every band the tile carries.
        dtype_override: Torch dtype to cast the loaded tensor to, if it
            should differ from the loaded dtype.
        key_pattern: Regex to extract the sample key from each file's name.
            None strips the extension and uses the rest — see `extract_key`.
        split: Text file of file stems to keep, one per line. None keeps
            every matching file found under `root`.

    Raises:
        ValueError: `extension` isn't one of `GEO_RASTER_EXTENSIONS`.
    """

    def __init__(
        self,
        root: str | Path,
        extension: GeoRasterExtension = ".tif",
        *,
        layer_name: LayerName = "image",
        sel_bands: list[str] | None = None,
        dtype_override: torch.dtype | None = None,
        key_pattern: str | None = None,
        split: str | Path | None = None,
    ) -> None:
        if extension not in GEO_RASTER_EXTENSIONS:
            raise ValueError(f"extension must be one of {GEO_RASTER_EXTENSIONS}, got {extension!r}")

        self.split = split
        self.root = Path(root)
        self.extension = extension
        self.layer_name = layer_name
        self.sel_bands = sel_bands
        self.dtype_override = dtype_override
        self.key_pattern = key_pattern
        self._cache: dict[str, GeoTile] = {}

        paths: dict[str, Path] = {
            extract_key(p.name, key_pattern): p for p in sorted(self.root.rglob(f"*{extension}"))
        }
        self.paths = filter_by_split(paths, split)
        self.reindex(self.paths)

    def _tile(self, key: str) -> GeoTile:
        if key not in self._cache:
            path = self.paths[key]
            self._cache[key] = GeoTile.from_zarr(path) if self.extension == ".zarr" else GeoTile.from_geotiff(path)
        return self._cache[key]

    def render(self, key: str) -> dict[str, Any]:
        """Render one sample. Lazily loads and caches the `GeoTile` for `key`.

        Args:
            key: File stem — one of `self.keys`.

        Returns:
            `{layer_name: Tensor, "anchor": GeoAnchor}` — bare anchor, no
            pixel data, always present.
        """
        tile = self._tile(key)
        tensor = tile.to_tensor(self.sel_bands)
        if self.dtype_override is not None:
            tensor = tensor.to(self.dtype_override)
        return {self.layer_name: tensor, "anchor": tile.to_anchor()}

    def to_row(self, key: str) -> dict[str, Any]:
        """Manifest row for `key`.

        Args:
            key: File stem — one of `self.keys`.

        Returns:
            `{"path": ...}` — `self.paths[key]` relative to `self.root`.
        """
        return {"path": str(self.paths[key].relative_to(self.root))}
