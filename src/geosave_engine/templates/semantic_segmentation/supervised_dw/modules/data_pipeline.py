from __future__ import annotations
from typing import Any

import numpy as np
from pathlib import Path
from pydantic import TypeAdapter
from scipy.ndimage import binary_opening

from geosave_engine.geodata.core import GeoTile, Pipeline, AnyIngestSource, GeotiffSource, ZarrSource
from geosave_engine.geodata.core import remap as remap_tile
from geosave_engine.geodata.algorithms import (
    compute_s2c_mask,
    compute_cdi_mask,
    compute_b10_mask,
    build_shadow_mask,
    compute_ndvi,
)
from geosave_engine.geodata.stac import StacClient

stac_client = StacClient.cdse()

_source_adapter: TypeAdapter[list[AnyIngestSource]] = TypeAdapter(list[AnyIngestSource])


class Sentinel2Pipeline(Pipeline):
    """Fetch all Sentinel-2 L1C bands for one anchor tile.

    Saves full 13-band set so downstream pipelines (cloud mask, NDVI)
    can read directly from zarr without re-fetching.
    """

    layer_name = "sentinel_2_l1c"
    resolution = 10
    description = "Sentinel-2 L1C imagery (all bands)"
    schema = [
        {"id": "B01", "name": "coastal"},
        {"id": "B02", "name": "blue"},
        {"id": "B03", "name": "green"},
        {"id": "B04", "name": "red"},
        {"id": "B05", "name": "rededge1"},
        {"id": "B06", "name": "rededge2"},
        {"id": "B07", "name": "rededge3"},
        {"id": "B08", "name": "nir"},
        {"id": "B09", "name": "watervapor"},
        {"id": "B10", "name": "cirrus"},
        {"id": "B11", "name": "swir16"},
        {"id": "B12", "name": "swir22"},
        {"id": "B8A", "name": "nir_narrow"},
    ]
    bands = [item["id"] for item in schema]
    source = stac_client.source(
        "sentinel-2-l1c",
        slot_mode="daily",
        composite="nearest",
        max_nodata_fraction=0.1,
    )

    def ingest(self, anchor: GeoTile) -> GeoTile:
        sentinel_2 = self.source.load(anchor, bands=self.bands)
        ds = sentinel_2.data
        if ds is None:
            raise ValueError(f"Sentinel-2 data missing for anchor at {anchor.centroid}")
        return sentinel_2.with_data(ds.astype(np.float32))


class CloudMaskPipeline(Pipeline):
    """Derive cloud and shadow mask from an ingested Sentinel-2 zarr tile."""

    layer_name = "cloud_mask"
    resolution = 10
    description = "Cloud and shadow mask"
    schema = [
        {"id": 0, "name": "clear",        "color": "#ffffff"},
        {"id": 1, "name": "cloud_shadow", "color": "#000000"},
    ]

    def ingest(self, anchor: GeoTile) -> GeoTile:
        """Compute cloud + shadow mask from anchor loaded via ingest_from_zarr."""
        ds = anchor.data
        if ds is None:
            raise ValueError(f"No data in anchor at {anchor.centroid}")
        if not anchor.stac:
            raise ValueError(f"STAC metadata missing at {anchor.centroid}")
        item = anchor.stac[0]

        s2c = compute_s2c_mask(
            b01=ds.sel(band="B01").values, b02=ds.sel(band="B02").values,
            b04=ds.sel(band="B04").values, b05=ds.sel(band="B05").values,
            b08=ds.sel(band="B08").values, b8a=ds.sel(band="B8A").values,
            b09=ds.sel(band="B09").values, b10=ds.sel(band="B10").values,
            b11=ds.sel(band="B11").values, b12=ds.sel(band="B12").values,
            prob_threshold=0.4,
        )
        cdi = compute_cdi_mask(
            b07=ds.sel(band="B07").values,
            b08=ds.sel(band="B08").values,
            b8a=ds.sel(band="B8A").values,
        )
        cirrus = compute_b10_mask(b10=ds.sel(band="B10").values, b10_threshold=0.0012)
        cloud_mask = binary_opening(s2c & cdi & cirrus, structure=np.ones((3, 3)))

        sun_az = item.properties.get("view:sun_azimuth", 0.0)
        shadow_mask = build_shadow_mask(
            cloud_mask, sun_azimuth_deg=sun_az, resolution=int(anchor.resolution)
        )
        return anchor.with_np((cloud_mask | shadow_mask).astype(np.uint8), ["cloud_mask"])


class NdviPipeline(Pipeline):
    """Derive NDVI from an ingested Sentinel-2 zarr tile."""

    layer_name = "ndvi"
    resolution = 10
    description = "Normalized Difference Vegetation Index"

    def ingest(self, anchor: GeoTile) -> GeoTile:
        ds = anchor.data
        if ds is None:
            raise ValueError(f"No data in anchor at {anchor.centroid}")
        ndvi = compute_ndvi(nir=ds.sel(band="B08").values, red=ds.sel(band="B04").values)
        return anchor.with_np(ndvi.astype(np.float32), ["ndvi"])


class LabelPipeline(Pipeline):
    """Remap DynamicWorld labels to contiguous class IDs.

    Classes 0, 9, 10 mapped to nodata (255). Remaining 1–8 remapped to 0–7.
    """

    layer_name = "dynamicworld"
    resolution = 10
    description = "DynamicWorld land cover labels (remapped)"
    nodata = 255
    schema = [
        {"id": 0, "name": "water",              "color": "#419bdf"},
        {"id": 1, "name": "trees",              "color": "#397d49"},
        {"id": 2, "name": "grass",              "color": "#88b053"},
        {"id": 3, "name": "flooded_vegetation", "color": "#7a87c6"},
        {"id": 4, "name": "crops",              "color": "#e49635"},
        {"id": 5, "name": "shrub_and_scrub",    "color": "#dfc35a"},
        {"id": 6, "name": "built",              "color": "#c4281b"},
        {"id": 7, "name": "bare",               "color": "#a59b8f"},
    ]
    # DynamicWorld Tier 1 → contiguous class ID (255 = nodata)
    # src: 0=No data, 1=Water, 2=Trees, 3=Grass, 4=Flooded Veg,
    #      5=Crops, 6=Scrub, 7=Built Area, 8=Bare Ground, 9=Snow/Ice, 10=Cloud
    _remap = {
        1: 0,    # Water          → 0
        2: 1,    # Trees          → 1
        3: 2,    # Grass          → 2
        4: 3,    # Flooded Veg    → 3
        5: 4,    # Crops          → 4
        6: 5,    # Scrub          → 5
        7: 6,    # Built Area     → 6
        8: 7,    # Bare Ground    → 7
        0: 255,  # No data        → nodata
        9: 255,  # Snow/Ice       → nodata
        10: 255, # Cloud          → nodata
    }

    @classmethod
    def class_map(cls) -> dict[int, str]:
        """Return ``{id: name}`` from schema."""
        return {int(item["id"]): str(item["name"]) for item in cls.schema if "name" in item}

    @classmethod
    def color_map(cls) -> dict[int, str]:
        """Return ``{id: color}`` from schema."""
        return {int(item["id"]): str(item["color"]) for item in cls.schema if "color" in item}

    def ingest(self, anchor: GeoTile) -> GeoTile:
        """Apply remap to anchor. Keeps first band only if multi-band input."""
        if anchor.data is not None and anchor.num_bands > 1:
            first_band = anchor.bands[0]
            data = anchor.data.sel(band=[first_band]).assign_coords(band=[self.layer_name])
            anchor = anchor.with_data(data.astype("int64"))
        return remap_tile(anchor, self._remap)


def training_pipeline(root: str | Path, geotiff_src: str | Path, max_item: int | None = None) -> None:
    """Run full training ingestion: S2 → cloud mask → NDVI → labels.

    Args:
        root: Workspace root; layer subdirs created inside.
        geotiff_src: Directory of GeoTIFF files used as anchors for S2 and label pipelines.
        max_item: Cap on tiles per pipeline stage. None processes all.
    """
    root = Path(root)
    tiff = GeotiffSource(src=Path(geotiff_src))
    zarr = ZarrSource(src=root / Sentinel2Pipeline.layer_name)
    Sentinel2Pipeline(root).ingest_from(tiff, max_item=max_item)
    CloudMaskPipeline(root).ingest_from(zarr, max_item=max_item)
    NdviPipeline(root).ingest_from(zarr, max_item=max_item)
    LabelPipeline(root).ingest_from(tiff, max_item=max_item)


def predict_pipeline(root: str | Path, sources: list[dict], max_item: int | None = None) -> None:
    """Run predict ingestion: S2 → cloud mask → NDVI (no labels).

    Args:
        root: Workspace root; layer subdirs created inside.
        sources: List of source dicts, each with a ``"type"`` discriminator key.
        max_item: Cap on tiles per pipeline stage. None processes all.

    Raises:
        ValidationError: If any source dict is missing required fields or has unknown type.
    """
    root = Path(root)
    zarr = ZarrSource(src=root / Sentinel2Pipeline.layer_name)
    s2 = Sentinel2Pipeline(root)
    for src in _source_adapter.validate_python(sources):
        s2.ingest_from(src, max_item=max_item)
    CloudMaskPipeline(root).ingest_from(zarr, max_item=max_item)
    NdviPipeline(root).ingest_from(zarr, max_item=max_item)
