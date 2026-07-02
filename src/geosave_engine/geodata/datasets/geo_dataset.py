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
    """Georeferenced PyTorch dataset over lazy GeoTiles under a workspace root.

    Examples:
        >>> ds = GeoDataset("/workspace/train")
        >>> loader = DataLoader(ds, batch_size=4, collate_fn=stack_samples)
        >>> batch = next(iter(loader))  # {"s2": Tensor[4,C,H,W], "label": Tensor[4,1,H,W]}

        Subclass to remap output keys and select bands:

        >>> class SegDataset(GeoDataset):
        ...     output_key = {"sentinel2": "image", "cloud_mask": "mask"}
        ...     sel_bands = {"sentinel2": ["B02", "B03", "B04"]}
    """

    output_key: dict[LayerName, str | tuple[str, torch.dtype]] = {}
    sel_bands: dict[LayerName, list[str]] = {}

    def __init__(
        self,
        root: str | Path,
        *,
        sampler: GeoTileSampler | None = None,
        output_key: dict[LayerName, str | tuple[str, torch.dtype]] | None = None,
        sel_bands: dict[LayerName, list[str]] | None = None,
    ) -> None:
        """
        Args:
            root: Workspace root directory with per-layer subdirs.
            sampler: Index builder. Defaults to PreChippedSampler.
            output_key: Layer name → batch key, or ``(batch_key, dtype)`` to cast on load.
            sel_bands: Layer name → band names to keep; default is all bands.
        """
        self.root = Path(root)
        self.catalog = self._scan(self.root)
        self.sampler = sampler or PreChippedSampler()
        self.output_key = output_key if output_key is not None else self.__class__.output_key
        self.sel_bands = sel_bands if sel_bands is not None else self.__class__.sel_bands
        self.index: gpd.GeoDataFrame = self.sampler.build_index(self.catalog)
        if len(self.index) == 0:
            log.warning("Empty dataset: no co-located samples across layers")

    @property
    def layers(self) -> list[LayerName]:
        """Layer names, in catalog order."""
        return list(self.catalog)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Render one sample.

        Args:
            index: Row index into the sample index.

        Returns:
            {
                "<output_key>": torch.Tensor,  # one entry per layer; tensors stacked if keys collide
                **extra_meta(...),
            }.
        """
        row = self.index.iloc[index]
        out: dict[str, Any] = {}
        tiles: dict[LayerName, GeoTile] = {}
        for layer in self.layers:
            tile: GeoTile = row[layer]
            tiles[layer] = tile
            entry = self.output_key.get(layer, layer)
            key, dtype = entry if isinstance(entry, tuple) else (entry, None)
            tensor = tile.to_tensor(self.sel_bands.get(layer), squeeze=True)
            if dtype is not None:
                tensor = tensor.to(dtype)
            out[key] = torch.cat([out[key], tensor], dim=0) if key in out else tensor

        out['context'] = self.context(tiles)
        
        return out

    def context(self, tiles: dict[LayerName, GeoTile]) -> dict[str, Any]:  # noqa: ARG002  # subclasses use tiles
        """Override to add per-sample metadata to the batch.

        Args:
            tiles: Rendered tiles for this sample, keyed by layer name.

        Returns:
            Dict merged into the sample. All values must be stackable by
            :func:`stack_samples` — tensors, numpy arrays, or Python scalars only.
            Python objects (e.g. GeoBox) will break the DataLoader.
        """
        return {}

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
