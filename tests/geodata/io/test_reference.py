"""Persisted window references use the same assets as whole-raster records."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from geosave_engine.geodata import read_vector


@pytest.mark.parametrize("georeferenced", [True, False])
def test_asset_reference_is_not_automatically_stac(tmp_path, georeferenced):
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {
            "id": ["tile"],
            "assets": [{"image": {"href": str(tmp_path / "image.nc")}}],
            "row_off": [0],
            "col_off": [0],
            "height": [1],
            "width": [1],
        },
        geometry=[box(0, 0, 1, 1) if georeferenced else None],
        crs="EPSG:4326" if georeferenced else None,
    )
    path = tmp_path / "reference.parquet"
    frame.gs.to_geoparquet(path)
    assert b"stac-geoparquet" not in pq.read_schema(path).metadata
    stored = pq.read_table(path).column("assets")[0].as_py()
    assert stored["image"]["href"] == "./image.nc"
    restored = read_vector(path)
    assert restored.crs == frame.crs
    assert restored.iloc[0].assets["image"]["href"] == str(tmp_path / "image.nc")


@pytest.mark.parametrize("format", ["netcdf", "zarr"])
def test_unindexed_pixel_raster_needs_no_geographic_id(tmp_path, format):
    import xarray as xr
    from geosave_engine.geodata import read_raster

    source = xr.Dataset({"image": (("y", "x"), np.arange(4).reshape(2, 2))})
    suffix = ".nc" if format == "netcdf" else ".zarr"
    saved = getattr(source.gs, f"to_{format}")(tmp_path / f"pixels{suffix}")
    with read_raster(saved) as restored:
        xr.testing.assert_equal(restored, source)
