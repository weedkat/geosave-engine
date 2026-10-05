"""Selected Series rows reopen raster assets without a catalog ID protocol."""

from pathlib import Path
import shutil

from dask.callbacks import Callback
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geosave_engine.geodata import GeoVector, read_vector
from tests.geodata.conftest import build_raster


def _written(root, raster, name):
    path = root / name
    return raster.isel(time=0).gs.to_cog(path)


def _sample_table(root: Path) -> gpd.GeoDataFrame:
    """Write two samples of two layers each and return their manifest."""
    rows = []
    for name in ("a", "b"):
        raster = build_raster(times=1)
        optical = _written(root, raster[["red"]], f"{name}/optical.tif")
        label = _written(root, raster[["nir"]], f"{name}/label.tif")
        rows.append(
            GeoVector.from_assets({"optical": optical, "label": label}, id=name)
        )
    return GeoVector.concat(rows).set_index("id", drop=False)


def test_a_row_opens_as_a_lazy_stack_of_its_assets(tmp_path: Path) -> None:
    table = _sample_table(tmp_path)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        sample = table.loc["b"].gs.to_xarray()

    assert sample.gs.groups == ("optical", "label")
    assert sample.gs.rasters["optical"].gs.variables == ("red",)
    assert started == []


def test_layers_names_the_assets_to_open(tmp_path: Path) -> None:
    table = _sample_table(tmp_path)

    assert table.loc["a"].gs.to_xarray(layers=["label"]).gs.groups == ("label",)
    with pytest.raises(KeyError, match="thermal"):
        table.loc["a"].gs.to_xarray(layers=["thermal"])


def test_native_selection_identifies_the_asset_row(tmp_path: Path) -> None:
    table = _sample_table(tmp_path)
    assert table.iloc[0].gs.to_xarray().gs.groups == ("optical", "label")
    with pytest.raises(KeyError):
        table.loc["missing"]


def test_one_layer_may_be_named_by_a_bare_string(tmp_path: Path) -> None:
    table = _sample_table(tmp_path)

    assert table.loc["a"].gs.to_xarray(layers="label").gs.groups == ("label",)


def test_a_layer_parquet_padded_with_null_is_not_a_layer(tmp_path: Path) -> None:
    raster = build_raster(times=1)
    both = GeoVector.from_assets(
        {
            "optical": _written(tmp_path, raster[["red"]], "a/optical.tif"),
            "label": _written(tmp_path, raster[["nir"]], "a/label.tif"),
        },
        id="a",
    )
    optical_only = GeoVector.from_assets(
        {"optical": _written(tmp_path, raster[["red"]], "b/optical.tif")}, id="b"
    )
    path = GeoVector.concat([both, optical_only]).gs.to_geoparquet(
        tmp_path / "manifest.parquet"
    )
    # Parquet pads the layers a row lacks with nulls, which read_vector removes.
    padded = gpd.read_parquet(path)

    with pytest.raises(KeyError, match=r"its layers are \['optical'\]"):
        padded.set_index("id").loc["b"].gs.to_xarray(layers=["label"])


def test_a_row_with_no_assets_names_no_rasters() -> None:
    table = gpd.GeoDataFrame(
        {"id": ["a"], "assets": [None]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )

    with pytest.raises(KeyError, match="names no assets"):
        table.iloc[0].gs.to_xarray()


def test_a_table_without_assets_names_no_rasters() -> None:
    with pytest.raises(KeyError, match="assets"):
        GeoVector.from_geometry(Point(0, 0)).iloc[0].gs.to_xarray()


def test_a_moved_manifest_still_opens_its_samples(tmp_path: Path) -> None:
    source = tmp_path / "prepared"
    _sample_table(source).gs.to_geoparquet(source / "manifest.parquet")
    moved = tmp_path / "moved"
    shutil.move(source, moved)

    sample = (
        read_vector(moved / "manifest.parquet").set_index("id").loc["a"].gs.to_xarray()
    )

    # Parquet stores the assets as one struct, which orders its keys by name.
    assert sample.gs.groups == ("label", "optical")


def test_a_geojson_row_needs_neither_id_nor_custom_index(tmp_path):
    image = build_raster(times=1).isel(time=0)
    saved = image.gs.to_cog(tmp_path / "image.tif")
    source = gpd.GeoDataFrame(
        {"name": ["field"], "assets": [{"image": {"href": str(saved)}}]},
        geometry=[Point(12, 45)],
        crs="EPSG:4326",
    )
    source.to_file(tmp_path / "fields.geojson", driver="GeoJSON")
    catalog = read_vector(tmp_path / "fields.geojson")

    row = catalog.loc[catalog.name == "field"].iloc[0]
    actual = row.gs.to_xarray()["image"].red
    np.testing.assert_array_equal(actual.values, image.red.values)
    assert actual.chunks is not None


def test_a_regular_row_does_not_apply_a_null_window():
    raster = build_raster(times=1)
    row = pd.Series({"row_off": None})
    assert row.gs.crop(raster) is raster


@pytest.mark.parametrize("window", ["valid", "invalid"])
def test_windowed_row_closes_the_assets_it_opened(monkeypatch, window):
    from geosave_engine.geodata import io

    closed = []

    def open_raster(href, **options):
        raster = build_raster()
        raster.set_close(lambda: closed.append(href))
        return raster

    monkeypatch.setattr(io, "read_raster", open_raster)
    fields = {
        "assets": {"image": {"href": "image.tif"}},
        "row_off": 0,
        "col_off": 0,
        "height": 1,
        "width": 1,
    }
    if window == "invalid":
        fields["row_off"] = "invalid"
    row = pd.Series(fields)
    if window == "invalid":
        with pytest.raises(ValueError):
            row.gs.to_xarray()
    else:
        with row.gs.to_xarray() as tile:
            assert tile["image"].red.shape == (1, 1)
            assert closed == []
    assert closed == ["image.tif"]
