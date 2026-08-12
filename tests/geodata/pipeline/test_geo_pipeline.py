"""Unit tests for GeoPipeline: ingest/ingest_to_tensor.

No network — a toy in-memory Pipeline stands in for a real STAC-backed one.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator

import numpy as np
import torch
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.spatial import GeoAnchor, GeoTag, GeoTile, GeoStack
from geosave_engine.geodata.pipeline import GeoPipeline

UTM = "EPSG:32633"


class _ToyPipeline(GeoPipeline):
    def fetch(self, anchor: GeoAnchor) -> Iterator[dict[str, GeoTile]]:
        arr = np.ones((1, 2, anchor.height, anchor.width), dtype="uint16")
        tile = anchor.to_geotile(arr, ["B02", "B03"], times=[anchor.start]).rebase(
            metadata={"description": "toy image"}
        )
        yield {"image": tile}


def _toy_anchor() -> GeoAnchor:
    d = datetime(2023, 2, 1)
    return GeoAnchor(
        geobox=GeoBox.from_bbox((500000, 5000000, 500320, 5000320), crs=UTM, resolution=10, anchor="edge"),
        geotag=GeoTag(datetime=(d, d)),
    )


class TestIngest:
    def test_yields_one_geostack_per_anchor(self):
        stacks = list(_ToyPipeline().ingest(_toy_anchor()))
        assert len(stacks) == 1
        assert isinstance(stacks[0], GeoStack)
        # standardized (time, band, y, x) — time=1 since temporal_slots defaults to 1
        assert tuple(stacks[0].tiles["image"].data.shape) == (1, 2, 32, 32)


class TestIngestToTensor:
    def test_renders_sample(self):
        anchor = _toy_anchor()
        samples = list(
            _ToyPipeline().ingest_to_tensor(
                anchor,
                sel_bands={"image": ["B02"]},
                dtype_override={"image": torch.float32},
            )
        )

        assert len(samples) == 1
        assert tuple(samples[0]["image"].shape) == (1, 1, 32, 32)  # time=1, C(sel B02), H, W
        assert samples[0]["image"].dtype == torch.float32
        start = datetime.fromisoformat(samples[0]["geotags"]["image"]["datetime"][0])
        assert start.isoformat(timespec="seconds") == "2023-02-01T00:00:00"

    def test_no_files_written(self, tmp_path):
        """ingest_to_tensor never touches disk — nothing to assert against tmp_path except that it stays empty."""
        list(_ToyPipeline().ingest_to_tensor(_toy_anchor()))
        assert list(tmp_path.iterdir()) == []
