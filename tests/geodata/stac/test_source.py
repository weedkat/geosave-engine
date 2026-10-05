from __future__ import annotations

from datetime import datetime as dt
from typing import Any

import dask.array as da
import numpy as np
import pytest
from pydantic import ValidationError

from geosave_engine.geodata.attrs import AttrsHeader
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave_engine.geodata.stac.source as source_module
from geosave_engine.geodata import GeoAnchor, raster
from geosave_engine.geodata.stac import StacQuery, StacSource, StacSourceConfig


class FakeClient:
    def search(self, query: object) -> list[object]:
        return [object()]

    def collection(self, collection: str) -> object:
        return object()


def _target() -> GeoAnchor:
    return GeoAnchor.from_coordinates(
        -6.5914,
        107.8416,
        shape=2,
        resolution=10,
        timespan="2025-06",
    )


def test_search_query_preserves_explicit_recipe_selectors() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]
    query = StacQuery(
        collections=["example"],
        intersects={"type": "Point", "coordinates": [107.8, -6.5]},
        datetime="2024-01",
    )
    source.query = query

    assert source._search_query(_target()) == query


def test_search_query_preserves_explicit_bbox() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]
    source.query = StacQuery(collections=["example"], bbox=(1, 2, 3, 4))

    query = source._search_query(_target())

    assert query.bbox == (1, 2, 3, 4)
    assert query.datetime == _target().timespan


def test_search_query_fills_omitted_target_selectors() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]

    query = source._search_query(_target())

    assert query.bbox is not None
    assert query.datetime == _target().timespan


def test_item_ids_do_not_gain_target_selectors() -> None:
    source = StacSource(FakeClient(), collection="example")  # type: ignore[arg-type]
    query = StacQuery(collections=["example"], ids=["scene-1"])
    source.query = query

    assert source._search_query(_target()) == query


def test_query_rejects_bbox_and_intersects_together() -> None:
    with pytest.raises(ValueError, match="bbox.*intersects"):
        StacQuery(
            collections=["example"],
            bbox=(0, 0, 1, 1),
            intersects={"type": "Point", "coordinates": [0, 0]},
        )


def test_source_config_defaults_to_spatial_dask_chunks() -> None:
    first = StacSourceConfig()
    second = StacSourceConfig()

    assert first.chunks == {"x": 1024, "y": 1024}
    assert first.chunks is not second.chunks


def test_source_config_accepts_mixed_chunk_sizes() -> None:
    configured = StacSourceConfig(chunks={"x": 1024, "y": "auto"})

    assert configured.chunks == {"x": 1024, "y": "auto"}


def test_source_config_keeps_one_kernel_per_band() -> None:
    configured = StacSourceConfig(resampling={"B04": "bilinear", "SCL": "nearest"})

    assert configured.resampling == {"B04": "bilinear", "SCL": "nearest"}


def test_source_config_refuses_a_kernel_nested_under_a_band() -> None:
    with pytest.raises(ValidationError, match="resampling"):
        StacSourceConfig(resampling={"B04": {"x": "nearest"}})


def test_source_config_refuses_an_unknown_kernel() -> None:
    with pytest.raises(ValidationError, match="resampling"):
        StacSourceConfig(resampling="not-a-kernel")


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
    monkeypatch.setattr(
        source_module, "create_header", lambda *args, **kwargs: AttrsHeader()
    )
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
    assert source.config.to_load_kwargs()["chunks"] is None


def test_with_properties_forwards_odc_native_requests() -> None:
    source = StacSource(FakeClient(), collection="example").set_config(  # type: ignore[arg-type]
        with_properties=(
            "platform",
            {
                "key": "view:sun_azimuth",
                "name": "sun_azimuth",
                "dtype": "float32",
                "units": "degree",
            },
        )
    )

    assert source.config.to_load_kwargs()["with_properties"] == [
        "platform",
        {
            "key": "view:sun_azimuth",
            "name": "sun_azimuth",
            "dtype": "float32",
            "units": "degree",
        },
    ]


def test_effective_nodata_follows_loaded_pixels(local_source):
    source, anchor = local_source
    loaded = source.set_config(dtype="float32", nodata=-9999).load(anchor)

    assert isinstance(loaded.red.data, da.Array)
    assert loaded.red.values[0, 0, 0] == -9999
    assert loaded.red.attrs["_FillValue"] == -9999
    prepared = loaded.gs.to_nan().gs.unpack()
    assert np.isnan(prepared.red.values[0, 0, 0])
    assert prepared.red.values[0, 1, 1] == pytest.approx(0.2)


def test_aliases_resolve_each_multiband_asset_index(local_source):
    source, anchor = local_source
    loaded = source.set_config(chunks=None).load(anchor)

    assert isinstance(loaded.red.data, np.ndarray)
    assert loaded.red.attrs["scale_factor"] == 0.0001
    assert loaded.nir.attrs["scale_factor"] == 0.0002
    assert loaded.nir.attrs["add_offset"] == -0.1
    assert loaded.gs.geobox == anchor.geobox
    assert loaded.gs.to_nan().gs.unpack().nir.values[0, 1, 1] == pytest.approx(1.1)
