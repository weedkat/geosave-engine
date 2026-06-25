from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import torch
from shapely.geometry import box
from torch.utils.data import Dataset

from geosave_engine.geodata.core import GeoTile
from geosave_engine.geodata.datasets.samplers import GeoTileSampler, PreChippedSampler

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
LayerName = str


class GeoDataset(Dataset):
    """A georeferenced PyTorch dataset over a per-layer catalog of lazy tiles.

    The catalog maps each layer name to a GeoDataFrame of that layer's tiles
    (columns ``[geometry, tile]``, WGS84 footprints). A :class:`GeoTileSampler`
    joins the catalog into a sample-row index. :meth:`__getitem__` returns one
    rendered sample: ``{output_key: tensor, ..., "geobox": GeoBox, "datetime": dt}``.
    :attr:`collate_fn` stacks a batch of those dicts into batched tensors.

    Subclasses may set ``output_key``/``sel_bands`` as class defaults; either can
    also be passed per-instance to override.

    Args:
        catalog: ``{layer: GeoDataFrame[geometry, tile]}``.
        sampler: Index builder. Defaults to :class:`PreChippedSampler`.
        output_key: Layer name → tensor key in the batch (default: the layer name).
        sel_bands: Layer name → band names to keep (default: all bands).
    """

    output_key: dict[LayerName, str] = {}
    sel_bands: dict[LayerName, list[str]] = {}

    def __init__(
        self,
        catalog: dict[LayerName, gpd.GeoDataFrame],
        *,
        sampler: GeoTileSampler | None = None,
        output_key: dict[LayerName, str] | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
    ) -> None:
        self.catalog = catalog
        self.sampler = sampler or PreChippedSampler()
        self.output_key = output_key if output_key is not None else self.__class__.output_key
        self.sel_bands = sel_bands if sel_bands is not None else self.__class__.sel_bands
        self.index: gpd.GeoDataFrame = self.sampler.build_index(catalog)
        if len(self.index) == 0:
            log.warning("Empty dataset: no co-located samples across layers")

    @property
    def layers(self) -> list[LayerName]:
        """Layer names, in catalog order."""
        return list(self.catalog)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Render one sample → ``{output_key: tensor, ...}`` plus any ``extra_meta``."""
        row = self.index.iloc[index]
        out: dict[str, Any] = {}
        tiles: dict[LayerName, GeoTile] = {}
        for layer in self.layers:
            tile: GeoTile = row[layer]
            tiles[layer] = tile
            key = self.output_key.get(layer, layer)
            tensor = tile.to_tensor(self.sel_bands.get(layer))
            out[key] = torch.cat([out[key], tensor], dim=0) if key in out else tensor
        out.update(self.extra_meta(tiles))
        return out

    def extra_meta(self, tiles: dict[LayerName, GeoTile]) -> dict[str, Any]:  # noqa: ARG002  # subclasses use tiles
        """Override to add per-sample metadata to the batch as tensors or arrays.

        All returned values must be stackable by ``stack_samples`` — use tensors,
        numpy arrays, or Python scalars. Python objects (e.g. GeoTile, GeoBox) will
        not stack and will confuse the DataLoader.
        """
        return {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(
        cls,
        root: str | Path,
        *,
        sampler: GeoTileSampler | None = None,
        output_key: dict[LayerName, str] | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
    ) -> "GeoDataset":
        """Build by scanning ``root/<layer>/*`` tile files into the catalog.

        Tiles open header-only (lazy); the files are the source of truth — no
        manifest is read.
        """
        return cls(
            cls._scan(Path(root)),
            sampler=sampler,
            output_key=output_key,
            sel_bands=sel_bands,
        )

    @classmethod
    def _scan(cls, root: Path) -> dict[LayerName, gpd.GeoDataFrame]:
        """Scan each ``root/<layer>/`` subdir into a per-layer GeoDataFrame."""
        catalog: dict[LayerName, gpd.GeoDataFrame] = {}
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            frame = cls._scan_layer(sub)
            if len(frame):
                catalog[sub.name] = frame
        return catalog

    @classmethod
    def _scan_layer(cls, layer_dir: Path) -> gpd.GeoDataFrame:
        """Open every tile header in ``layer_dir`` → ``[geometry, tile]`` frame."""
        tiles = [
            GeoTile.from_zarr(p, load_data=False)
            for p in sorted(layer_dir.glob("*.zarr"))
        ]
        for pattern in ("*.tif", "*.tiff"):
            tiles += [
                GeoTile.from_geotiff(p, load_data=False)
                for p in sorted(layer_dir.glob(pattern))
            ]
        geometry = [box(*t.wgs84_bbox) for t in tiles]
        return gpd.GeoDataFrame(
            {"tile": tiles, "geometry": geometry}, geometry="geometry", crs=WGS84
        )
