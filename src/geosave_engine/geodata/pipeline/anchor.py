import rasterio
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime as dt
from affine import Affine
from odc.geo.geobox import GeoBox
from rasterio.warp import transform_bounds


DEFAULT_DATE_PATTERN: str = r"(?<!\d)(\d{8})(?!\d)"
DEFAULT_DATE_FORMAT:  str = "%Y%m%d"

@dataclass(frozen=True)
class Anchor:
    """To be the anchor for blueprints, containing the necessary information to define the area of interest and other parameters."""
    affine: Affine
    crs: str
    width: int
    height: int
    datetime: dt

    @property
    def resolution(self) -> float:
        return self.affine.a

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Returns the bounding box in WGS84 (EPSG:4326) for STAC compatibility."""
        bb = self.to_geobox().boundingbox
        return transform_bounds(self.crs, "EPSG:4326", bb.left, bb.bottom, bb.right, bb.top)

    @classmethod
    def from_tiff(cls, path: str | Path, date_format: str = DEFAULT_DATE_FORMAT, date_pattern: str = DEFAULT_DATE_PATTERN) -> "Anchor":
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

        return cls(affine=affine, crs=crs, width=width, height=height, datetime=datetime)
    
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