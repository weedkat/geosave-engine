from __future__ import annotations

from geosave_engine.geodata.core import GeoTile, Pipeline


class ImagePipeline(Pipeline):
    """Fetch or derive image data for one anchor tile.

    Replace with your data source logic.
    """

    layer_name = "image"
    resolution = 10
    description = "Input imagery"
    schema: list[dict] = []  # define bands: [{"id": "B01", "name": "blue"}, ...]

    def ingest(self, anchor: GeoTile) -> GeoTile:
        raise NotImplementedError("Implement ingest() to load image data for this anchor.")


class LabelPipeline(Pipeline):
    """Derive or load segmentation labels for one anchor tile.

    Replace with your label source logic.
    """

    layer_name = "label"
    resolution = 10
    description = "Segmentation labels"
    nodata = 255
    schema: list[dict] = []  # define classes: [{"id": 0, "name": "class_a", "color": "#ff0000"}, ...]

    def ingest(self, anchor: GeoTile) -> GeoTile:
        raise NotImplementedError("Implement ingest() to load label data for this anchor.")
