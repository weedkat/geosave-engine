from __future__ import annotations

import numpy as np
from pathlib import Path

from geosave_engine.geodata.tile import GeoTile
from geosave_engine.geodata.pipeline import GeoPipeline, GeotiffSource, save_dataset


class HLSS30Pipeline(GeoPipeline):
    """Ingest 6-band HLS S30 tiles from pre-downloaded GeoTIFFs.

    Stores bands B02, B03, B04, B8A, B11, B12 as float32 DN values
    (surface reflectance × 10000), matching GraniteGeospatialBiomass input.

    GeoTIFFs must have band coordinates named B02, B03, B04, B8A, B11, B12.
    """

    bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]

    def context(self, tiles: dict[str, GeoTile]) -> dict[str, object]:
        """Spatial metadata passed through to predict_step output.

        Returns:
            {
                "crs": str,
                "transform": Affine,
                "coordinate": tuple[float, float],
            }
        """
        ref = next(iter(tiles.values()))
        return {"crs": ref.crs, "transform": ref.affine, "coordinate": ref.centroid}

    def fetch(self, anchor: GeoTile) -> dict[str, GeoTile]:
        """Anchor already carries its data (GeotiffSource loaded it) — no I/O left."""
        return {"hls_s30": anchor}

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        tile = raw["hls_s30"]
        return {"hls_s30": tile.with_data(tile.data.sel(band=self.bands).astype(np.float32))}


def predict_pipeline(
    root: str | Path,
    geotiff_src: str | Path,
    max_item: int | None = None,
) -> None:
    """Ingest HLS S30 tiles from GeoTIFF source for prediction.

    Args:
        root: Workspace root; one anchor subdirectory created inside.
        geotiff_src: Directory of 6-band HLS S30 GeoTIFFs in DN scale.
            Filename stems must end in ``-YYYYMMDD`` or
            ``-YYYYMMDD-YYYYMMDD``.
        max_item: Cap on tiles. None processes all.
    """
    root = Path(root)
    tiff = GeotiffSource(src=Path(geotiff_src))
    save_dataset(HLSS30Pipeline(), tiff.to_anchors(limit=max_item), root)
