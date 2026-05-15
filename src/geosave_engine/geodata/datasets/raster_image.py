from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from pyproj import CRS
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import Sample


class RasterImage(RasterDataset):
    """Generic multi-band image layer as a TorchGeo ``RasterDataset``.

    Serves any multi-band GeoTIFF produced by the geosave ingest pipeline,
    laid out as ``<root>/<split>/*.tif``. Returns samples under key ``"image"``.
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

    def __init__(
        self,
        paths: Path | Iterable[Path],
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        time_series: bool = False,
        filename_regex: str | None = None,
        date_format: str | None = None,
    ) -> None:
        if filename_regex is not None:
            self.filename_regex = filename_regex
        if date_format is not None:
            self.date_format = date_format
        super().__init__(paths, crs, res, bands, transforms, cache, time_series)
