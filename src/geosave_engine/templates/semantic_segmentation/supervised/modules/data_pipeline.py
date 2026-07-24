from __future__ import annotations

from functools import cached_property

from geosave_engine.geodata.tile import GeoTile
from geosave_engine.geodata.pipeline import GeoPipeline
from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.stac.source import StacSource

L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B09", "B10", "B11", "B12", "B8A",
]
INPUT_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]


class Pipeline(GeoPipeline):
    """Sentinel-2 imagery + cloud/shadow mask + NDVI for one anchor.

    Imagery only — DynamicWorld label prep is a separate, project-specific
    step (see ``workspace/scripts/ingest.py``), attached to each sample this
    pipeline's ``ingest()`` yields before that script's own single save. Not
    this class's concern: label remapping doesn't generalize across
    projects the way STAC-driven imagery ingest does.
    """
    @cached_property
    def sources(self) -> dict[str, StacSource]:
        """Built lazily on first real fetch — importing/instantiating Pipeline
        alone must not cost a live STAC network call."""
        stac_client = StacClient.cdse()
        # temporal_slots=1 (scene granularity, StacSource default) — preprocess()
        # below assumes exactly one time step per raw sample, no loop.
        return {
            "sentinel_2_l1c": stac_client.source(
                "sentinel-2-l1c", bands=L1C_BANDS, max_nodata_fraction=0.1, temporal_slots=1
            )
        }

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        """Derive final layers (masks, indices, ...) from fetched raw layers.

        Default: passthrough. See docs/concept/pipeline.md for the
        alignment/timing model.
        """
        return raw
