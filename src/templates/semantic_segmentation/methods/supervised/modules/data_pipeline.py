from __future__ import annotations
from typing import Any

import numpy as np
from scipy.ndimage import binary_opening

from geosave_engine.geodata.core import GeoTile, Pipeline, remap
from geosave_engine.geodata.source import Source
from geosave_engine.geodata.algorithms import (
    compute_s2c_mask,
    compute_cdi_mask,
    compute_b10_mask,
    build_shadow_mask,
    compute_ndvi,
)
from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.datasets import GeoDataset

stac_client = StacClient.cdse()

class DataPipeline(Pipeline):
    layer_schema = {
        "sentinel_2_l1c": {
            "resolution": 10,
            "description": "Sentinel-2 L1C imagery",
            "bands": {
                "B02": {"name": "blue"},
                "B03": {"name": "green"},
                "B04": {"name": "red"},
                "B05": {"name": "rededge1"},
                "B06": {"name": "rededge2"},
                "B07": {"name": "rededge3"},
                "B08": {"name": "nir"},
                "B11": {"name": "swir16"},
                "B12": {"name": "swir22"},
            },
        },
        "cloud_mask": {
            "resolution": 10,
            "description": "Cloud and shadow mask",
            "classes": {
                0: {"name": "clear", "color": "#ffffff"},
                1: {"name": "cloud_shadow", "color": "#000000"},
            },
        },
        "ndvi": {
            "resolution": 10,
            "description": "Normalized Difference Vegetation Index",
            "bands": {},
        },
    }

    s2_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
    selected_bands = list(layer_schema["sentinel_2_l1c"]["bands"].keys())
    source = Source.sentinel_2_l1c(client=stac_client, max_nodata_fraction=0.1)

    @classmethod
    def band_map(cls) -> dict[int | str, str]:
        """Return ``{band_code: name}`` from layer_schema."""
        return cls.get_meta_map("sentinel_2_l1c", "name")

    @classmethod
    def cloud_color_map(cls) -> dict[int | str, str]:
        """Return ``{class_id: color}`` for cloud mask."""
        return cls.get_meta_map("cloud_mask", "color")
    @classmethod
    def cloud_class_map(cls) -> dict[int | str, str]:
        """Return ``{class_id: name}`` for cloud mask."""
        return cls.get_meta_map("cloud_mask", "name")

    def ingest(self, anchor: GeoTile) -> dict[str, GeoTile]:
        sentinel_2: GeoTile = self.source.load(anchor, reduce="nearest", bands=list(self.s2_bands))

        ds = sentinel_2.data
        if ds is None:
            raise ValueError(f"Sentinel-2 data missing for anchor at {anchor.centroid}")
        if not sentinel_2.stac:
            raise ValueError(f"STAC metadata missing for Sentinel-2 data at {anchor.centroid}")
        item = sentinel_2.stac[0]

        s2c = compute_s2c_mask(
            b01=ds["B01"].values, b02=ds["B02"].values,
            b04=ds["B04"].values, b05=ds["B05"].values,
            b08=ds["B08"].values, b8a=ds["B8A"].values,
            b09=ds["B09"].values, b10=ds["B10"].values,
            b11=ds["B11"].values, b12=ds["B12"].values,
            prob_threshold=0.4,
        )
        cdi = compute_cdi_mask(
            b07=ds["B07"].values,
            b08=ds["B08"].values,
            b8a=ds["B8A"].values,
        )
        cirrus = compute_b10_mask(b10=ds["B10"].values, b10_threshold=0.0012)

        cloud_mask = s2c & cdi & cirrus
        cloud_mask = binary_opening(cloud_mask, structure=np.ones((3, 3)))

        sun_az = item.properties.get("view:sun_azimuth", 0.0)
        shadow_mask = build_shadow_mask(
            cloud_mask, sun_azimuth_deg=sun_az, resolution=int(anchor.resolution)
        )

        ndvi = compute_ndvi(nir=ds["B08"].values, red=ds["B04"].values)

        return {
            "sentinel_2_l1c": anchor.with_data(ds[self.selected_bands].astype(np.float32)),
            "cloud_mask": anchor.with_np((cloud_mask | shadow_mask).astype(np.uint8), ["cloud_mask"]),
            "ndvi": anchor.with_np(ndvi.astype(np.float32), ["ndvi"]),
        }

class LabelPipeline(Pipeline):
    remap = {
        0: 255,   # nodata → ignore
        1: 0,     # water
        2: 1,     # trees
        3: 2,     # grass
        4: 3,     # flooded_vegetation
        5: 4,     # crops
        6: 5,     # shrub_and_scrub
        7: 6,     # built
        8: 7,     # bare
        9: 255,   # snow_and_ice → ignore
        10: 255,  # cloud → ignore
    }
    layer_schema = {
        "dynamicworld": {
            "resolution": 10,
            "description": "DynamicWorld land cover labels (remapped)",
            "classes": {
                0: {"name": "water", "color": "#419bdf"},
                1: {"name": "trees", "color": "#397d49"},
                2: {"name": "grass", "color": "#88b053"},
                3: {"name": "flooded_vegetation", "color": "#7a87c6"},
                4: {"name": "crops", "color": "#e49635"},
                5: {"name": "shrub_and_scrub", "color": "#dfc35a"},
                6: {"name": "built", "color": "#c4281b"},
                7: {"name": "bare", "color": "#a59b8f"},
                255: {"name": "ignore", "color": "#000000"},
            },
        },
    }

    @classmethod
    def class_map(cls) -> dict[int, str]:
        """Return ``{class_id: name}`` from layer_schema."""
        return cls.get_meta_map("dynamicworld", "name")

    @classmethod
    def color_map(cls) -> dict[int, str]:
        """Return ``{class_id: color}`` from layer_schema."""
        return cls.get_meta_map("dynamicworld", "color")

    def ingest(self, anchor: GeoTile) -> dict[str, GeoTile]:
        # Multi-band anchor: first variable = ground truth label, rest = class probabilities
        if anchor.data is not None and anchor.num_bands > 1:
            first = next(iter(anchor.data.data_vars))
            anchor = anchor.with_data(anchor.data[[first]].rename({first: "label"}))
        return {"dynamicworld": remap(anchor, self.remap)}


class DynamicWorldDataset(GeoDataset):
    output_key = {
        "sentinel_2_l1c": "image",
        "cloud_mask": "mask",
        "ndvi": "ndvi",
        "dynamicworld": "label",
    }

    def extra_meta(self, tiles: dict[str, GeoTile]) -> dict[str, Any]:
        """Override to add per-sample metadata to the batch as tensors or arrays.

        All returned values must be stackable by ``stack_samples`` — use tensors,
        arrays, or scalars. For example, you can return the following:
        """
        meta = {}
        ref_tile = next(iter(tiles.values()))
        meta["crs"] = ref_tile.crs
        meta["centroid"] = ref_tile.centroid
        meta["affine"] = ref_tile.affine
        meta["day"] = ref_tile.datetime.timetuple().tm_yday
        return meta

class DynamicWorldDatasetRGB(DynamicWorldDataset):
    output_key = {
        "sentinel_2_l1c": "image",
        "cloud_mask": "mask",
        "ndvi": "ndvi",
        "dynamicworld": "label",
    }
    sel_bands = {
        "sentinel_2_l1c": ["B04", "B03", "B02"],
    }
