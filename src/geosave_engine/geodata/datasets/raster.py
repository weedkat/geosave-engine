"""RasterDataset: PyTorch dataset over standalone raster files. See RasterDataset for details."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from torch.utils.data import Dataset

from geosave_engine.geodata.spatial import ContextFn, GeoRaster, GeoTile, TensorTile
from geosave_engine.geodata.utils.io import READERS

if TYPE_CHECKING:
    from geosave_engine.geodata.extensions import TilerMode


class RasterDataset(Dataset[TensorTile]):
    """PyTorch dataset over standalone raster files, one sample per tile.

    Reads every format `GeoRaster.open` knows — GeoTIFF, COG, Zarr, NetCDF.
    Tiles are cut once and held lazily; no pixel is read until `__getitem__`.
    A file this configuration can't cut is warned about and skipped.

    Args:
        root: Directory holding raster files, searched recursively.
        bands: Band names to keep, in this order. Selected at open, so
            dropped bands are never read. None keeps every band.
        dtype: Torch dtype to cast to. None keeps each file's stored dtype.
        tile_size_px: Window side length in pixels. None uses the shorter
            axis, so each file yields one whole-surface sample.
        stride_px: Distance between window origins. None = `tile_size_px`.
        overlap: Forwarded to the tiler. Wins over `stride_px`.
        mode: How a trailing window's overhang is filled — "reflect",
            "edge", or "constant", which needs a declared nodata.
        vector: True gives each tile its file's features, filtered to the window.
        name: Extra text folded into each cut's derived `group_id`.
        context_fn: Called once per window while the index is built; its
            result becomes that tile's `model_context`.
        suffixes: File suffixes to index. None indexes every suffix
            `GeoRaster.open` supports.

    Raises:
        ValueError: `root` isn't a directory, `suffixes` names a suffix with
            no reader, or no file under `root` yielded a tile.

    Examples:
        >>> dataset = RasterDataset("data/tiles", bands=["B04", "B03", "B02"], tile_size_px=512)
        >>> dataset[0]["data"].shape
        torch.Size([3, 512, 512])
    """

    def __init__(
        self,
        root: str | Path,
        bands: list[str] | None = None,
        dtype: torch.dtype | None = None,
        *,
        tile_size_px: int | None = None,
        stride_px: int | None = None,
        overlap: int | float | tuple[int, int] | None = None,
        mode: TilerMode = "reflect",
        vector: bool = True,
        name: str | None = None,
        context_fn: ContextFn | None = None,
        suffixes: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"RasterDataset needs a directory of raster files, got {self.root}")
        indexed = tuple(READERS) if suffixes is None else tuple(suffixes)
        unknown = sorted(suffix for suffix in indexed if suffix.lower() not in READERS)
        if unknown:
            raise ValueError(f"no reader for suffix(es) {unknown} — known: {sorted(READERS)}")

        self.bands = bands
        self.dtype = dtype
        self.tile_size_px = tile_size_px
        self.stride_px = stride_px
        self.overlap = overlap
        self.mode: TilerMode = mode
        self.vector = vector
        self.name = name
        self.context_fn = context_fn

        wanted = {suffix.lower() for suffix in indexed}
        self.tiles: list[GeoTile] = []
        for path in sorted(p for p in self.root.rglob("*") if p.suffix.lower() in wanted):
            try:
                raster = GeoRaster.open(path, bands=None if bands is None else tuple(bands))
                self.tiles.extend(
                    raster.tiles(
                        tile_size_px=tile_size_px,
                        stride_px=stride_px,
                        overlap=overlap,
                        mode=mode,
                        vector=vector,
                        name=name,
                        context_fn=context_fn,
                    )
                )
            except (ValueError, KeyError) as error:
                warnings.warn(f"skipping {path}: {type(error).__name__}: {error}", stacklevel=2)
        if not self.tiles:
            raise ValueError(f"no raster under {self.root} with suffix in {sorted(indexed)} yielded a tile")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.root}, samples={len(self)}, bands={self.bands})"

    def __len__(self) -> int:
        """Number of tiles in the index.

        Returns:
            How many samples this dataset serves.
        """
        return len(self.tiles)

    def __getitem__(self, index: int) -> TensorTile:
        """Read one tile into a tensor sample.

        Args:
            index: Position in the index.

        Returns:
            {
                "data": torch.Tensor,  # (band, y, x) or (time, band, y, x)
                "anchor": GeoAnchor,  # this window's own grid, with its vector
                "model_context": {
                    "<key>": torch.Tensor | str | None,
                },
            }
        """
        return self.tiles[index].to_sample(dtype=self.dtype)
