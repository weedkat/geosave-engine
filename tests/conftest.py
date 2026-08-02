from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from dotenv import load_dotenv
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros

from geosave_engine.geodata.tile import GeoTag, GeoTile as GeoLayer


def pytest_configure(config):
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


_DEFAULT_BBOX = (-22.749, 15.979, -22.5, 16.1)
_DEFAULT_RES = 0.001
_DEFAULT_DT = datetime(2019, 2, 23)


@pytest.fixture
def make_geo_layer():
    """Factory producing synthetic GeoLayer instances for tests."""

    def _make(
        bbox: tuple = _DEFAULT_BBOX,
        resolution: float = _DEFAULT_RES,
        dt: datetime = _DEFAULT_DT,
        with_data: bool = False,
        bands: int = 1,
        band_names: list[str] | None = None,
        times: list[str] | None = None,
        fill: float = 0.0,
        dtype: str = "float32",
    ) -> GeoLayer:
        """Build a tile whose ``data`` is a Dataset of single-band variables.

        Each variable is shaped ``(y, x)`` (or ``(time, y, x)`` when ``times`` is
        given) — the one-band-per-variable pattern GeoTile expects.
        """
        gbox = GeoBox.from_bbox(bbox, crs="EPSG:4326", resolution=resolution, anchor="edge")
        data = None
        if with_data:
            names = band_names or [f"band_{i}" for i in range(bands)]
            base = xr_zeros(gbox, dtype=dtype) + fill
            vars_ = {}
            for name in names:
                da = base
                if times is not None:
                    da = da.expand_dims(time=[np.datetime64(t) for t in times]).copy()
                vars_[name] = da
            data = xr.Dataset(vars_).rio.write_crs("EPSG:4326")
        return GeoLayer(geobox=gbox, geotag=GeoTag(datetime=(dt, dt)), data=data)

    return _make


@pytest.fixture
def dw_tif_path() -> Path:
    """Path to the DynamicWorld anchor used as a real test fixture."""
    return Path(__file__).parent / "data" / "dw_-22.7491991582_15.9791703445-20190223.tif"
