import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.core.stack import stack as build_stack


def build_raster(
    *,
    crs: str = "EPSG:32749",
    times: int = 0,
    packed: bool = False,
) -> xr.Dataset:
    """Build one raster the way an adapter would.

    Args:
        crs: CRS of the two-by-two grid.
        times: Length of the time axis, or 0 for a timeless raster.
        packed: Give each variable CF packing over stored digital numbers.

    Returns:
        Raster Dataset holding `red` and `nir` as uint16.

    Examples:
        >>> build_raster(times=2).red.dims
        ('time', 'y', 'x')
    """
    geobox = GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_020.0, 9_000_020.0),
        crs=crs,
        shape=(2, 2),
        tight=True,
    )
    coords = dict(xr_coords(geobox, always_yx=True))
    spatial_dims = ("y", "x")
    values = np.array([[1000, 2000], [3000, 0]], dtype="uint16")
    if times:
        coords["time"] = xr.DataArray(
            np.array(
                [f"2025-06-{day + 1:02d}" for day in range(times)],
                dtype="datetime64[ns]",
            ),
            dims="time",
        )
        block = np.broadcast_to(values, (times, 2, 2)).copy()
        dims = ("time", *spatial_dims)
    else:
        block = values
        dims = spatial_dims

    raw = xr.Dataset({"red": (dims, block), "nir": (dims, block)}, coords=coords)
    if packed:
        for name in raw.data_vars:
            raw[name].attrs.update(
                {
                    "scale_factor": np.float32(1e-4),
                    "add_offset": np.float32(0.0),
                    "_FillValue": np.uint16(0),
                }
            )
    return raw.gs.write_crs()


@pytest.fixture
def raster() -> xr.Dataset:
    return build_raster()


@pytest.fixture
def stack(raster: xr.Dataset) -> xr.DataTree:
    return build_stack({"optical": raster[["red"]], "infrared": raster[["nir"]]})


@pytest.fixture
def bucket():
    """Yield a unique `memory://` prefix and empty it afterwards."""
    import uuid

    import fsspec

    filesystem = fsspec.filesystem("memory")
    prefix = f"memory://geosave-tests/{uuid.uuid4()}"
    yield prefix
    if filesystem.exists(prefix):
        filesystem.rm(prefix, recursive=True)


def whole_windows(parents):
    """List rasters held in memory as whole windows, as `cuts.stacks` lists saved ones.

    Args:
        parents: Window id mapped to a DataArray, Dataset or stack. A lone
            raster is the group `image`.

    Returns:
        One window per parent, stating each group's time labels as they are
        and the parent's grid, or no grid for unreferenced pixels.
    """
    import geopandas as gpd
    import numpy as np
    import xarray as xr
    from odc.geo.geobox import GeoBox

    rows = []
    for name, parent in parents.items():
        rasters = (
            parent.gs.rasters if isinstance(parent, xr.DataTree) else {"image": parent}
        )
        times = {}
        for group, each in rasters.items():
            stamps = each.coords.get("time")
            if stamps is None:
                times[group] = None
            elif np.issubdtype(stamps.dtype, np.datetime64):
                labels = np.atleast_1d(stamps.values).astype("datetime64[us]")
                times[group] = np.datetime_as_string(labels).tolist()
            else:
                times[group] = list(np.atleast_1d(stamps.values))
        first = next(iter(rasters.values()))
        grid = first.gs.geobox
        located = isinstance(grid, GeoBox) and grid.crs is not None
        rows.append(
            {
                "id": name,
                "parent": None,
                "stack": name,
                "times": times,
                "start_datetime": None,
                "end_datetime": None,
                "crs": str(grid.crs) if located else None,
                "transform": list(grid.transform)[:6]
                if located
                else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "row_off": 0,
                "col_off": 0,
                "height": first.sizes["y"],
                "width": first.sizes["x"],
                "geometry": grid.extent.to_crs("EPSG:4326").geom if located else None,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
