from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from pyproj import CRS
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import Sample


class Sentinel2L1C(RasterDataset):
    """Sentinel-2 Level-1C (Top-of-Atmosphere reflectance) as a TorchGeo ``RasterDataset``.

    Expects multiband GeoTIFFs produced by the geosave ingestion pipeline, laid
    out as ``<root>/<split>/*.tif``. Default filename convention is
    ``<prefix>_<lon>_<lat>-<YYYYMMDD>.tif`` (e.g.
    ``dw_-61.1513923557_4.5757852347-20190223.tif``). Pass ``filename_regex``
    and/or ``date_format`` to support alternative conventions.

    Band order inside each TIFF is assumed to match ``all_bands`` unless the
    caller supplies ``bands`` to restrict/reorder.
    """
    filename_glob = "*.tif"
    filename_regex = r"""
        ^(?P<prefix>[A-Za-z0-9_]+?)
        _(?P<lon>-?\d+(?:\.\d+)?)
        _(?P<lat>-?\d+(?:\.\d+)?)
        -(?P<date>\d{8})
        \.tif$
    """
    date_format = "%Y%m%d"

    is_image = True
    separate_files = False

    # https://sentiwiki.copernicus.eu/web/s2-mission
    all_bands: tuple[str, ...] = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    )

    resolutions: ClassVar[dict[str, str]] = {
        "B01": "60m",
        "B02": "10m",
        "B03": "10m",
        "B04": "10m",
        "B05": "20m",
        "B06": "20m",
        "B07": "20m",
        "B08": "10m",
        "B8A": "20m",
        "B09": "60m",
        "B10": "60m",
        "B11": "20m",
        "B12": "20m",
    }

    rgb_bands = ("B04", "B03", "B02")

    # Central wavelength (μm)
    wavelengths: ClassVar[dict[str, float]] = {
        "B01": 0.4427,
        "B02": 0.4927,
        "B03": 0.5598,
        "B04": 0.6646,
        "B05": 0.7041,
        "B06": 0.7405,
        "B07": 0.7828,
        "B08": 0.8328,
        "B8A": 0.8647,
        "B09": 0.9451,
        "B10": 1.3735,
        "B11": 1.6137,
        "B12": 2.2024,
        # For compatibility with other dataset naming conventions
        "B1": 0.4427,
        "B2": 0.4927,
        "B3": 0.5598,
        "B4": 0.6646,
        "B5": 0.7041,
        "B6": 0.7405,
        "B7": 0.7828,
        "B8": 0.8328,
        "B9": 0.9451,
    }

    def __init__(
        self,
        paths: Path | Iterable[Path],
        crs: CRS | None = None,
        res: float | tuple[float, float] = 10,
        bands: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        time_series: bool = False,
        filename_regex: str | None = None,
        date_format: str | None = None,
    ) -> None:
        # Torchgeo file discovery reads these as instance attributes during
        # __init__, so overrides must be set before calling super().
        if filename_regex is not None:
            self.filename_regex = filename_regex
        if date_format is not None:
            self.date_format = date_format
        super().__init__(paths, crs, res, bands, transforms, cache, time_series)
