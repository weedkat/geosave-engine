from __future__ import annotations

import dataclasses
import numpy as np
from scipy.ndimage import binary_opening

from geosave_engine.geodata.algorithms import (
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_ndvi,
    compute_s2c_mask,
)
from geosave_engine.geodata.core import GeoTile, Pipeline
from geosave_engine.geodata.core import remap as remap_tile
from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.stac.query import StacQuery

_stac_client = StacClient.cdse()


class Sentinel2Pipeline(Pipeline):
    """Fetch all 13 Sentinel-2 L1C bands for one anchor tile.

    Saves full band set so downstream pipelines (cloud mask, NDVI)
    can read from zarr without re-fetching.
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
    source = _stac_client.source(
        "sentinel-2-l1c",
        slot_mode="daily",
        composite="nearest",
        max_nodata_fraction=0.1,
    )

    def ingest(self, anchor: GeoTile) -> GeoTile:
        tile = self.source.load(anchor, bands=self.bands)
        if tile.data is None:
            raise ValueError(f"Sentinel-2 data missing for anchor at {anchor.centroid}")
        return tile.with_data(tile.data.astype(np.float32))

    def _expand_anchor(self, anchor: GeoTile) -> list[GeoTile]:
        if not isinstance(anchor.datetime, tuple):
            return [anchor]
        start, end = anchor.datetime
        query = StacQuery(
            collections=[self.source.collection_id],
            bbox=anchor.wgs84_bbox,
            datetime=(start, end),
        )
        items = self.source.client.search(query)
        if not items:
            raise ValueError(
                f"No Sentinel-2 scenes found for bbox={anchor.wgs84_bbox} "
                f"between {start.date()} and {end.date()}"
            )
        return [
            dataclasses.replace(anchor, datetime=item.datetime)
            for item in items
            if item.datetime is not None
        ]


class Sentinel2RGBPipeline(Sentinel2Pipeline):
    """Fetch only B04/B03/B02 from Sentinel-2 L1C. Faster ingest, no cloud mask or NDVI."""

    description = "Sentinel-2 L1C RGB bands (B04, B03, B02)"
    schema = [
        {"id": "B04", "name": "red"},
        {"id": "B03", "name": "green"},
        {"id": "B02", "name": "blue"},
    ]
    bands = ["B04", "B03", "B02"]


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
    # src: 0=No data, 1=Water, 2=Trees, 3=Grass, 4=Flooded Veg,
    #      5=Crops, 6=Scrub, 7=Built Area, 8=Bare Ground, 9=Snow/Ice, 10=Cloud
    _remap = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
        0: 255, 9: 255, 10: 255,
    }

    def ingest(self, anchor: GeoTile) -> GeoTile:
        if anchor.data is not None and anchor.num_bands > 1:
            first_band = anchor.bands[0]
            data = anchor.data.sel(band=[first_band]).assign_coords(band=[self.layer_name])
            anchor = anchor.with_data(data.astype("int64"))
        return remap_tile(anchor, self._remap)
