from __future__ import annotations

from typing import Any

import torch

from geosave_engine.geodata.core import GeoTile
from geosave_engine.geodata.datasets.geo_dataset import GeoDataset


class DynamicWorldDataset(GeoDataset):
    """GeoDataset with full DynamicWorld output mapping and spatial context.

    Expects layers: sentinel_2_l1c, cloud_mask, ndvi, dynamicworld.
    """

    output_key = {
        "sentinel_2_l1c": "image",
        "cloud_mask":      ("mask",  torch.bool),
        "ndvi":            ("ndvi",  torch.float32),
        "dynamicworld":    ("label", torch.int64),
    }

    def context(self, tiles: dict[str, GeoTile]) -> dict[str, Any]:
        """Add spatial and temporal metadata to each batch sample.

        Returns:
            {
                "crs": str,
                "transform": Affine,
                "coordinate": tuple[float, float],
                "time": int,           # day of year (1-365)
                "datetime": str,       # ISO 8601 datetime string
                "bbox_wgs84": list,    # [minlon, minlat, maxlon, maxlat]
                "stac_item_ids": list, # STAC item IDs for provenance
            }.
        """
        ref = next(iter(tiles.values()))
        ref_dt = ref.datetime[0] if isinstance(ref.datetime, tuple) else ref.datetime
        return {
            "crs": ref.crs,
            "transform": ref.affine,
            "coordinate": ref.centroid,
            "time": ref_dt.timetuple().tm_yday,
            "datetime": ref_dt.isoformat(),
            "bbox_wgs84": list(ref.wgs84_bbox),
            "stac_item_ids": [item.id for item in ref.stac],
        }


class DynamicWorldRGBDataset(DynamicWorldDataset):
    """RGB-only DynamicWorld dataset. Expects layers: sentinel_2_l1c, dynamicworld.

    No cloud mask or NDVI — use with Sentinel2RGBPipeline for fast ingest.
    Selects B04/B03/B02 from the full S2 zarr if all bands were ingested.
    """

    output_key = {
        "sentinel_2_l1c": "image",
        "dynamicworld":    ("label", torch.int64),
    }
    sel_bands: dict = {"sentinel_2_l1c": ["B04", "B03", "B02"]}
