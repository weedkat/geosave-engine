import re
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path

import rasterio
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from rasterio.warp import transform_bounds


DEFAULT_DATE_PATTERN: str = r"(?<!\d)(\d{8})(?!\d)"
DEFAULT_DATE_FORMAT:  str = "%Y%m%d"
ANCHOR_CACHE_KEY: str = "__anchor__"

@dataclass(frozen=True)
class Anchor:
    """To be the anchor for blueprints, containing the necessary information to define the area of interest and other parameters."""
    affine: Affine
    crs: str
    width: int
    height: int
    datetime: dt
    label: xr.DataArray | None = field(default=None, compare=False)

    @property
    def resolution(self) -> float:
        return self.affine.a

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Returns the bounding box in WGS84 (EPSG:4326) for STAC compatibility."""
        bb = self.to_geobox().boundingbox
        return transform_bounds(self.crs, "EPSG:4326", bb.left, bb.bottom, bb.right, bb.top)

    @classmethod
    def from_tiff(
        cls,
        path: str | Path,
        date_format: str = DEFAULT_DATE_FORMAT,
        date_pattern: str = DEFAULT_DATE_PATTERN,
        load_label: bool = True,
        select_band: int = 0,
    ) -> "Anchor":
        path = Path(path)
        with rasterio.open(path) as src:
            affine = src.transform
            crs = src.crs.to_string()
            height, width = src.shape
            date_str = src.tags().get("datetime")

        if not date_str:
            match = re.search(date_pattern, path.name)
            if not match:
                raise ValueError(f"No date found in TIFF filename '{path.name}'")
            date_str = match.group(1)

        datetime = dt.strptime(date_str, date_format)

        label: xr.DataArray | None = None
        if load_label:
            da_raw = rioxarray.open_rasterio(path)
            if not isinstance(da_raw, xr.DataArray):
                raise TypeError(f"Expected DataArray from {path}, got {type(da_raw)}")
            label = da_raw.isel(band=select_band, drop=True)

        return cls(affine=affine, crs=crs, width=width, height=height, datetime=datetime, label=label)
    
    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        crs: str, 
        resolution: float,
        datetime: dt
    ) -> "Anchor":
        """Create an anchor from a bounding box and resolution."""
        
        # We use GeoBox to 'discover' the correct affine and shape
        # anchor="edge" ensures pixels snap to the coordinate grid (like Sentinel-2)
        gbox = GeoBox.from_bbox(
            bbox, 
            crs=crs, 
            resolution=resolution, 
            anchor="edge"
        )
        
        return cls(
            affine=gbox.affine,
            crs=str(gbox.crs),
            width=gbox.width,
            height=gbox.height,
            datetime=datetime
        )
    
    def to_geobox(self) -> GeoBox:
        """This is what you pass to odc-stac."""
        return GeoBox(shape=(self.height, self.width), affine=self.affine, crs=self.crs)