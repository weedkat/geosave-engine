"""Geographic metadata shared by written encoder context functions."""

import pandas as pd
from affine import Affine
from odc.geo.geobox import GeoBox


def geographic_center(row: pd.Series) -> tuple[float, float]:
    """Return latitude and longitude from the row's exact raster grid.

    Args:
        row: Window row stating its grid as `crs`, `transform`, `height`
            and `width`.

    Returns:
        Geographic centre in latitude/longitude order.

    Raises:
        ValueError: The sample has no georeferenced grid.
    """
    crs = row.get("crs")
    if not isinstance(crs, str):
        raise ValueError("model location context requires a georeferenced grid")
    grid = GeoBox(
        (int(row["height"]), int(row["width"])), Affine(*row["transform"]), crs
    )
    longitude, latitude = grid.extent.centroid.to_crs("EPSG:4326").coords[0]
    return float(latitude), float(longitude)
