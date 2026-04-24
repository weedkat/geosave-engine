from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from pyproj import CRS
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import Sample


class RasterLabel(RasterDataset):
    """Generic single-band raster label as a TorchGeo ``RasterDataset``.

    Serves any raster label product (DynamicWorld, ESA WorldCover, etc.)
    already ingested into the geosave workspace layout
    (``<root>/<split>/*.tif``). Default filename convention is
    ``<prefix>_<lon>_<lat>-<YYYYMMDD>.tif``; override ``filename_regex`` and/or
    ``date_format`` per product when needed.

    Paired with an imagery dataset (e.g. ``Sentinel2L1C``) via TorchGeo's ``&``
    operator to yield aligned ``{image, mask}`` samples.
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

    is_image = False
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
