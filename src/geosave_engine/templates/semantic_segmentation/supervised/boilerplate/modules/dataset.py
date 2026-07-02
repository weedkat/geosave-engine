from __future__ import annotations

from typing import Any

import torch

from geosave_engine.geodata.core import GeoTile
from geosave_engine.geodata.datasets.geo_dataset import GeoDataset


class WorkspaceDataset(GeoDataset):
    """GeoDataset mapping ingested layers to model input keys.

    Update ``output_key`` to match your pipeline layer names.
    Each value is either a string key or ``(key, dtype)`` tuple.
    """

    output_key = {
        "image": "image",
        "label": ("label", torch.int64),
    }

    def context(self, tiles: dict[str, GeoTile]) -> dict[str, Any]:
        """Return per-sample spatial/temporal metadata added to each batch.

        Returns:
            {
                "crs": str,
                "transform": Affine,
                "coordinate": tuple[float, float],
                "time": int,
            }.
        """
        ref = next(iter(tiles.values()))
        return {
            "crs": ref.crs,
            "transform": ref.affine,
            "coordinate": ref.centroid,
            "time": ref.datetime.timetuple().tm_yday,
        }
