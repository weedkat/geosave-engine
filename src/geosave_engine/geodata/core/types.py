"""Parameter bundles for the geodata ingestion API."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadRequest:
    """Parameters for a single ``odc.stac.load`` invocation.

    ``crs`` and ``bounds`` pin the output to an exact pixel grid. When omitted,
    a UTM CRS is derived from ``bbox``.

    ``bands`` is required — callers must declare which bands they want; there is
    no silent "all bands" default.
    """

    bands: list[str]
    bbox: tuple[float, float, float, float] | None = None
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    resolution: int = 10
    as_reflectance: bool = True


@dataclass(frozen=True)
class GeoTiffOptions:
    """rasterio ``to_raster`` options for the ingestor's ``save`` method.

    ``predictor=None`` means the predictor is derived from the array dtype at
    save time (3 for floating-point, 2 for integer).
    """

    compress: str = "lzw"
    predictor: int | None = None
    tiled: bool = True
