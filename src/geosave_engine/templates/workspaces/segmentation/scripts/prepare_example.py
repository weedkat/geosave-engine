"""Prepare sample reflectance: python scripts/prepare_example.py."""

from pathlib import Path

import dask.array as da
from odc.geo.geobox import GeoBox
import xarray as xr

from geosave_engine.geodata.core.raster import raster
from geosave_engine.model.spec import ModelSpec


def sample_raster() -> xr.Dataset:
    """Return a small lazy raster with four packed Sentinel-2 bands."""
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    optical = raster(
        {
            name: da.full((4, 4), value, chunks=(2, 2), dtype="uint16")
            for name, value in (
                ("B02", 1000),
                ("B03", 1500),
                ("B04", 2000),
                ("B08", 6000),
            )
        },
        grid,
    )
    for variable in optical.data_vars.values():
        variable.attrs.update(scale_factor=0.0001, add_offset=0.0)
    return optical


def main() -> None:
    """Prepare lazy reflectance and compute only to display the sample pixels."""
    workspace = Path(__file__).resolve().parents[1]
    spec = ModelSpec.load(workspace / "configs/model_spec.yaml")
    results = spec.preprocess({"sentinel_2_l2a": sample_raster()})
    print(results["image"].compute())


if __name__ == "__main__":
    main()
