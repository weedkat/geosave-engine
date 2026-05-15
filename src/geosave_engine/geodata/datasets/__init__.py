"""TorchGeo-compatible dataset classes."""
from geosave_engine.geodata.datasets.raster_mask import RasterMask
from geosave_engine.geodata.datasets.raster_label import RasterLabel
from geosave_engine.geodata.datasets.raster_image import RasterImage

__all__ = ["RasterImage", "RasterLabel", "RasterMask"]
