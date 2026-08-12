"""MosaicStore: one contiguous georeferenced surface, written incrementally.

SKELETON — spec not settled, do not implement against this yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from odc.geo.geobox import GeoBox

from geosave_engine.geodata.spatial.ops import MergeMethod

if TYPE_CHECKING:
    from geosave_engine.geodata.spatial.tile import GeoTile

LayerName = str


class MosaicStore:
    """One big georeferenced surface (e.g. a prediction run's output), backed by zarr.

    Unlike mosaic_spatial/mosaic_stack (merge N in-memory tiles into one
    right now), this is a disk-backed store tiles get written into over
    time — e.g. as an inference job completes tile by tile — servable
    afterward (WMTS). Real zarr sharding fits here: chunk/shard boundaries
    are actual neighboring geography, unlike SampleStore's samples.

    Args:
        path: Zarr store root. Created if missing, opened for windowed
            write/read if present.
        geobox: Full extent + resolution + CRS this mosaic covers — fixed
            at creation, every write is a window into it.
        chunk_px: Spatial chunk side length.
        shard_px: Spatial shard side length. None skips sharding.
    """

    def __init__(
        self,
        path: str | Path,
        geobox: GeoBox,
        chunk_px: int = 512,
        shard_px: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.geobox = geobox
        raise NotImplementedError("spec not settled — see class docstring")

    def write(self, tile: GeoTile, layer: LayerName, method: MergeMethod = "first") -> None:
        """Write one tile into its windowed position in the mosaic.

        Plan: xr.Dataset.to_zarr's own `region=` param writes into a slice
        of a pre-existing array — real mechanism, not hand-rolled offset math.

        Args:
            tile: Must share this mosaic's CRS/resolution — no reprojection here.
            layer: Which named surface this tile belongs to (own zarr array).
            method: Overlap-resolution rule where tile overlaps existing pixels.

        Raises:
            ValueError: tile's CRS/resolution doesn't match, or tile falls
                outside geobox's extent.
        """
        raise NotImplementedError("spec not settled — see class docstring")

    def read(self, bbox: tuple[float, float, float, float], layer: LayerName) -> GeoTile:
        """Windowed read of one region back out as a GeoTile.

        Args:
            bbox: Region to read, in this mosaic's own CRS.
            layer: Which named surface to read.

        Raises:
            NotImplementedError: Always — skeleton only.
        """
        raise NotImplementedError("spec not settled — see class docstring")

    def close(self) -> None:
        """Flush any pending writes.

        Raises:
            NotImplementedError: Always — skeleton only.
        """
        raise NotImplementedError("spec not settled — see class docstring")
