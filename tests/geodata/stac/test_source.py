from __future__ import annotations

from datetime import datetime as dt
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.stac.source as source_module
from geosave_engine.geodata import GeoAnchor, raster
from geosave_engine.geodata.stac import StacSource, StacSourceConfig


class FakeClient:
    def search(self, query: object) -> list[object]:
        return [object()]

    def collection(self, collection: str) -> object:
        return object()


def test_source_config_defaults_to_spatial_dask_chunks() -> None:
    first = StacSourceConfig()
    second = StacSourceConfig()

    assert first.chunks == {"x": 1024, "y": 1024}
    assert first.chunks is not second.chunks


def test_source_config_accepts_mixed_chunk_sizes() -> None:
    configured = StacSourceConfig(chunks={"x": 1024, "y": "auto"})

    assert configured.chunks == {"x": 1024, "y": "auto"}


def test_load_returns_dask_backed_arrays_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_load(items: list[object], *, geobox: GeoBox, **options: Any) -> xr.Dataset:
        captured.update(options)
        values = np.ones((1, *geobox.shape), dtype="uint16")
        loaded = raster({"red": values}, geobox, time=[dt(2025, 6, 1)])
        return loaded.chunk(options["chunks"])

    monkeypatch.setattr(source_module.odc.stac, "load", fake_load)
    monkeypatch.setattr(source_module, "stamp_stac", lambda ds, *args, **kwargs: ds)
    anchor = GeoAnchor.from_coordinates(
        -6.5914,
        107.8416,
        shape=2,
        resolution=10,
        timespan="2025-06",
    )

    loaded = StacSource(FakeClient(), collection="example").load(anchor)  # type: ignore[arg-type]

    assert captured["chunks"] == {"x": 1024, "y": 1024}
    assert isinstance(loaded["red"].data, da.Array)


def test_chunks_none_remains_an_explicit_eager_mode() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]

    source.set_config(chunks=None)

    assert source.config.chunks is None
    assert source._load_options()["chunks"] is None
