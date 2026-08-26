from __future__ import annotations

import numpy as np

from geosave_engine.geodata.spatial import GeoRaster
from geosave_engine.geodata.features import compute_scl_mask
from geosave_engine.geodata.pipeline import GeoPipeline
from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.stac.source import StacSource

# All real L2A spectral bands (no B10 — TOA cirrus-only, dropped by
# atmospheric correction). Ingested in full — which subset a given model
# actually trains on is a `band_map` config decision (configs/metadata.yaml),
# not an ingest-time one. Want a leaner fetch that skips unneeded bands?
# Write a separate data_pipeline.py for that, don't narrow this one.
L2A_SPECTRAL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
L2A_BANDS = [*L2A_SPECTRAL_BANDS, "SCL"]

# ESA Sen2Cor's own SCL legend (Sen2Cor ATBD) — same class/color scheme SCL
# ships with everywhere (QGIS default style, ESA docs, etc).
SCL_CLASS_MAP = {
    0: "no_data", 1: "saturated_defective", 2: "dark_area", 3: "cloud_shadow",
    4: "vegetation", 5: "bare_soil", 6: "water", 7: "unclassified",
    8: "cloud_medium_prob", 9: "cloud_high_prob", 10: "thin_cirrus", 11: "snow_ice",
}
SCL_COLOR_MAP = {
    0: "#000000", 1: "#ff0000", 2: "#2f2f2f", 3: "#643200",
    4: "#00a600", 5: "#ffe65a", 6: "#0000ff", 7: "#808080",
    8: "#c0c0c0", 9: "#ffffff", 10: "#66ccff", 11: "#ff66ff",
}


class Pipeline(GeoPipeline):
    """Sentinel-2 L2A imagery + Scene Classification Layer + SCL-derived cloud mask for one anchor.

    Practical alternative to `data_pipeline_l1c.Pipeline`: L2A is already
    atmospherically corrected (surface reflectance, not TOA) and ships its
    own per-pixel Scene Classification Layer, so cloud/shadow/snow masking
    is a lookup (`compute_scl_mask`) instead of running s2cloudless/CDI/B10
    detectors — faster ingest, no L1C pixel artifacting, and Planetary
    Computer serves it with signed URLs, no CDSE credentials needed. Kept
    separate from the L1C pipeline (not a drop-in replacement) since that
    one exists specifically to reproduce the DynamicWorld paper's own L1C +
    s2cloudless/CDI/B10 methodology.
    """

    def __init__(self) -> None:
        self.client = StacClient.planetary_computer()

    def sources(self) -> dict[str, StacSource]:
        # groupby="time" keeps every acquisition its own step — DynamicWorld is per scene.
        return {
            "sentinel_2_l2a": self.client.source("sentinel-2-l2a").set_config(
                bands=L2A_BANDS,
                groupby="time",
            )
        }

    def preprocess(self, raw: dict[str, GeoRaster]) -> dict[str, GeoRaster]:
        s2 = raw["sentinel_2_l2a"]
        # one scene per sample (see `sources` above), so drop straight to (band, y, x)
        ds = s2.data.isel(time=0)

        scl = ds.sel(band="SCL").astype(np.uint8)
        mask = compute_scl_mask(scl).astype(np.uint8)

        s2_layer = s2.anchor.to_raster(
            ds.sel(band=L2A_SPECTRAL_BANDS),
            attrs=s2.attrs.rebase(
                tags={
                    **s2.tags,
                    "description": f"Sentinel-2 L2A surface reflectance ({len(L2A_SPECTRAL_BANDS)} bands)",
                },
                render={"rgb_bands": ("B04", "B03", "B02")},
            ),
        )

        scl_layer = s2.anchor.to_raster(
            scl,
            bands=["SCL"],
            attrs=s2.attrs.rebase(
                tags={
                    **s2.tags,
                    "description": "Sentinel-2 L2A Scene Classification Layer (Sen2Cor)",
                },
                render={"rgb_bands": None, "class_map": SCL_CLASS_MAP, "color_map": SCL_COLOR_MAP},
            ),
        )

        cloud_mask_layer = s2.anchor.to_raster(
            mask,
            bands=["cloud_mask"],
            attrs=s2.attrs.rebase(
                tags={
                    **s2.tags,
                    "description": "Cloud/shadow/invalid mask from Scene Classification Layer (SCL)",
                },
                render={
                    "rgb_bands": None,
                    "class_map": {0: "clear", 1: "cloud/shadow/invalid"},
                    "color_map": {0: "#FFFFFF", 1: "#000000"},
                },
            ),
        )

        # first key anchors the stack `ingest` builds
        return {
            "sentinel_2_l2a": s2_layer,
            "scl": scl_layer,
            "cloud_mask": cloud_mask_layer,
        }
