"""A row opens its assets, and applies its pixel window once."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dask.callbacks import Callback

from geosave_engine.geodata import GeoVector, io
from geosave_engine.geodata.stac import asset, item

from tests.geodata.conftest import build_raster


def _sample(tmp_path: Path) -> pd.Series:
    optical = build_raster(times=2).gs.to_zarr(tmp_path / "optical.zarr")
    (label,) = build_raster(times=1).isel(time=0)[["nir"]].gs.to_cog(tmp_path / "label")
    entry = item.from_assets(
        {"optical": asset.from_path(optical), "label": asset.from_path(label)}, id="s0"
    )
    return GeoVector.from_items([entry]).iloc[0]


def test_a_sample_row_opens_its_layers_as_a_lazy_stack(tmp_path: Path) -> None:
    row = _sample(tmp_path)
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        sample = row.gs.to_stack(layers=["optical", "label"])

    assert started == []
    assert sample.gs.groups == ("optical", "label")
    assert sample.gs.rasters["optical"].sizes["time"] == 2
    assert sample.gs.rasters["label"].gs.variables == ("nir",)
    # Without names the groups follow the table's asset order, which Parquet sorts.
    assert set(row.gs.to_stack().gs.groups) == {"optical", "label"}


def test_a_row_opens_one_layer_as_a_raster(tmp_path: Path) -> None:
    raster = _sample(tmp_path).gs.to_raster(layer="optical")

    assert raster.gs.variables == ("red", "nir")
    assert raster.red.chunks is not None


def test_a_missing_layer_is_a_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="dem"):
        _sample(tmp_path).gs.to_stack(layers=["dem"])


def test_a_windowed_row_reads_cropped_once(tmp_path: Path) -> None:
    row = pd.concat(
        [
            _sample(tmp_path),
            pd.Series({"row_off": 0, "col_off": 1, "height": 1, "width": 1}),
        ]
    )

    sample = row.gs.to_stack(layers="label")

    assert sample.gs.rasters["label"].sizes["x"] == 1
    assert sample.gs.rasters["label"].sizes["y"] == 1


def test_assets_that_are_not_data_are_left_alone(tmp_path: Path) -> None:
    path = io.geotiff.write_cog(
        build_raster(times=1).isel(time=0), tmp_path / "scene.tif"
    )
    row = pd.Series(
        {
            "assets": {
                "image": {"href": str(path), "roles": ["data"]},
                "thumbnail": {
                    "href": "https://example.com/t.jpg",
                    "roles": ["thumbnail"],
                },
                "absent": None,
            }
        }
    )

    assert row.gs.hrefs == {"image": str(path)}
    np.testing.assert_array_equal(row.gs.to_raster().red.squeeze(), build_raster().red)


def test_explicit_window_crops_native_pixels():
    data = build_raster(times=1).chunk()
    row = pd.Series({"row_off": 0, "col_off": 1, "height": 1, "width": 1})
    actual = row.gs.crop(data)
    assert actual.sizes["y"] == actual.sizes["x"] == 1
    assert actual.gs.geobox == data.gs.geobox.translate_pix(1, 0).crop((1, 1))
    assert actual.red.chunks is not None


def test_a_row_without_a_window_preserves_native_data():
    data = build_raster(times=1)
    assert pd.Series({"row_off": None}).gs.crop(data) is data
