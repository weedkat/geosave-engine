from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening

from geosave_engine.geodata.spatial import GeoRaster
from geosave_engine.geodata.features import (
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_ndvi,
    compute_s2c_mask,
)
from geosave_engine.geodata.pipeline import GeoPipeline
from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.stac.source import StacSource
from geosave_engine.geodata.utils.array import map_overlap

L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B09", "B10", "B11", "B12", "B8A",
]

# DynamicWorld paper's model input set: all bands except B01/B8A/B09/B10 —
# fetched anyway (L1C_BANDS above) since cloud-mask derivation needs them,
# but the saved sentinel_2_l1c layer only carries what the model actually
# takes. odc-stac already resamples every band onto one geobox (bilinear by
# default, matching the paper) at StacSource.load time — no separate
# upscaling step needed here.
DW_MODEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]


class Pipeline(GeoPipeline):
    """Sentinel-2 imagery + cloud/shadow mask + NDVI for one anchor.

    This pipeline is a concrete example of how to use GeoPipeline to build a
    multi-layer sample from a single anchor. It fetches Sentinel-2 L1C imagery
    from the Copernicus Data Space Ecosystem STAC endpoint, derives a cloud/shadow mask 
    and NDVI from it, and returns a dict of GeoTile layers.
    """

    def __init__(self) -> None:
        self.client = StacClient.cdse()

    def sources(self) -> dict[str, StacSource]:
        # groupby="time" keeps every acquisition its own step — DynamicWorld is per scene.
        return {
            "sentinel_2_l1c": self.client.source("sentinel-2-l1c").set_config(
                bands=L1C_BANDS,
                groupby="time",
            )
        }

    def preprocess(self, raw: dict[str, GeoRaster]) -> dict[str, GeoRaster]:
        s2 = raw["sentinel_2_l1c"]
        # one scene per sample (see `sources` above), so drop straight to (band, y, x)
        ds = s2.data.isel(time=0)
        records = s2.stac.at(s2.times[0], s2.timespec) if s2.stac else ()
        sun_az = records[0].properties.get("view:sun_azimuth", 0.0) if records else 0.0

        # s2cloudless is trained on reflectance, not the DN a provider publishes. This scale
        # ignores the -1000 radiometric offset baseline 04.00+ products carry; read
        # `s2:processing_baseline` off the item records if your collection needs it.
        toa = ds / 10_000

        # Cloud mask/NDVI derive from the full 13-band fetch (they need
        # B01/B8A/B09/B10) — select down to the model's own input bands
        # only after they're computed.
        s2c = compute_s2c_mask(
            b01=toa.sel(band="B01"), b02=toa.sel(band="B02"),
            b04=toa.sel(band="B04"), b05=toa.sel(band="B05"),
            b08=toa.sel(band="B08"), b8a=toa.sel(band="B8A"),
            b09=toa.sel(band="B09"), b10=toa.sel(band="B10"),
            b11=toa.sel(band="B11"), b12=toa.sel(band="B12"),
            prob_threshold=0.4,
        )
        cdi = compute_cdi_mask(
            b07=toa.sel(band="B07"),
            b08=toa.sel(band="B08"),
            b8a=toa.sel(band="B8A"),
        )
        cirrus = compute_b10_mask(b10=toa.sel(band="B10"), b10_threshold=0.0012)
        opened = map_overlap(
            lambda block: binary_opening(block, structure=np.ones((3, 3))),
            (s2c.astype(bool) & cdi & cirrus),
            depth=1,
            dtype="bool",
        )
        shadow = build_shadow_mask(opened, sun_azimuth_deg=sun_az, resolution=int(s2.resolution))
        mask = (opened | shadow).astype(np.uint8)

        ndvi = compute_ndvi(nir=toa.sel(band="B08"), red=toa.sel(band="B04"))

        s2_layer = s2.anchor.to_raster(
            ds.sel(band=DW_MODEL_BANDS),
            attrs=s2.attrs.rebase(
                tags={
                    **s2.tags,
                    "description": f"Sentinel-2 L1C imagery ({len(DW_MODEL_BANDS)} bands, DynamicWorld input set)",
                },
                render={"rgb_bands": ("B04", "B03", "B02")},
            ),
        )

        cloud_mask_layer = s2.anchor.to_raster(
            mask,
            bands=["cloud_mask"],
            attrs=s2.attrs.rebase(
                tags={**s2.tags, "description": "Cloud and shadow mask"},
                render={
                    "rgb_bands": None,
                    "class_map": {0: "clear", 1: "cloud/shadow"},
                    "color_map": {0: "#FFFFFF", 1: "#000000"},
                },
            ),
        )

        ndvi_layer = s2.anchor.to_raster(
            ndvi,
            bands=["ndvi"],
            attrs=s2.attrs.rebase(
                tags={**s2.tags, "description": "Normalized Difference Vegetation Index"},
                render=None,
            ),
        )

        # first key anchors the stack `ingest` builds
        return {
            "sentinel_2_l1c": s2_layer,
            "cloud_mask": cloud_mask_layer,
            "ndvi": ndvi_layer,
        }
