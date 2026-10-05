"""Geographic metadata shared by written encoder context functions."""

import pandas as pd
from affine import Affine
from odc.geo.geobox import GeoBox


def geographic_center(row: pd.Series) -> tuple[float, float]:
    """Return latitude and longitude from the row's exact raster grid.

    Args:
        row: Sample row with projection shape, transform and CRS fields.

    Returns:
        Geographic centre in latitude/longitude order.

    Raises:
        ValueError: The sample has no georeferenced grid.
    """
    code = row.get("proj:code")
    crs = code if isinstance(code, str) else row.get("proj:wkt2")
    if not isinstance(crs, str):
        raise ValueError("model location context requires a georeferenced grid")
    grid = GeoBox(tuple(row["proj:shape"]), Affine(*row["proj:transform"]), crs)
    longitude, latitude = grid.extent.centroid.to_crs("EPSG:4326").coords[0]
    return float(latitude), float(longitude)
