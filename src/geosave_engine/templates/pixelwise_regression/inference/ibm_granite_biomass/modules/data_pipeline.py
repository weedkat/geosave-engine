from __future__ import annotations

import numpy as np
from pathlib import Path

from geosave_engine.geodata.core import GeoTile, Pipeline, GeotiffSource


class HLSS30Pipeline(Pipeline):
    """Ingest 6-band HLS S30 tiles from pre-downloaded GeoTIFFs.

    Stores bands B02, B03, B04, B8A, B11, B12 as float32 DN values
    (surface reflectance × 10000), matching GraniteGeospatialBiomass input.

    GeoTIFFs must have band coordinates named B02, B03, B04, B8A, B11, B12.
    """

    layer_name = "hls_s30"
    resolution = 30
    description = "HLS S30 (Sentinel-2 derived) 6-band surface reflectance"
    schema = [
        {"id": "B02", "name": "blue"},
        {"id": "B03", "name": "green"},
        {"id": "B04", "name": "red"},
        {"id": "B8A", "name": "nir_narrow"},
        {"id": "B11", "name": "swir_1"},
        {"id": "B12", "name": "swir_2"},
    ]
    bands = [item["id"] for item in schema]

    def ingest(self, anchor: GeoTile) -> GeoTile:
        ds = anchor.data
        if ds is None:
            raise ValueError(f"No data in HLS anchor at {anchor.centroid}")
        return anchor.with_data(ds.sel(band=self.bands).astype(np.float32))


def predict_pipeline(
    root: str | Path,
    geotiff_src: str | Path,
    max_item: int | None = None,
) -> None:
    """Ingest HLS S30 tiles from GeoTIFF source for prediction.

    Args:
        root: Workspace root; layer subdirs created inside.
        geotiff_src: Directory of 6-band HLS S30 GeoTIFFs in DN scale.
        max_item: Cap on tiles. None processes all.
    """
    root = Path(root)
    tiff = GeotiffSource(src=Path(geotiff_src))
    HLSS30Pipeline(root).ingest_from(tiff, max_item=max_item)
