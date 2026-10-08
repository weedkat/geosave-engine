"""A window is read off the opened sample it was cut from."""

from __future__ import annotations

import dask
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata import cuts, raster
from geosave_engine.geodata.io import geoparquet

from tests.geodata.conftest import whole_windows


def _refuse(*args, **kwargs):
    raise AssertionError("reading a window computed pixels")


def test_a_chip_reads_its_instants_and_pixels(items, opened) -> None:
    windows = cuts.chips(cuts.frames(cuts.stacks(items), 2, tolerance="10D"), 32)
    window = windows.iloc[0]

    with dask.config.set(scheduler=_refuse):
        chip = cuts.select(opened, window)

    assert dict(chip["s2"].sizes) == {"time": 2, "y": 32, "x": 32}
    assert [str(label)[:10] for label in chip["s2"].time.values] == [
        "2024-03-08",
        "2024-03-28",
    ]
    assert dict(chip["dem"].sizes) == {"y": 32, "x": 32}
    whole = opened["s2"].dataset.sel(time=chip["s2"].time)
    np.testing.assert_array_equal(
        chip["s2"]["red"].values, whole["red"].values[:, :32, :32]
    )


def test_a_read_past_the_edge_is_filled_as_the_window_states(items, opened) -> None:
    windows = cuts.chips(cuts.stacks(items), 32, overlap=8)

    chip = cuts.select(opened, windows.iloc[0])

    assert dict(chip["label"].sizes) == {"y": 32, "x": 32}
    assert np.isfinite(chip["dem"]["height"].values).all()


def test_a_whole_window_reads_the_whole_sample(items, opened, grid) -> None:
    (window,) = cuts.stacks(items).to_dict("records")

    whole = cuts.select(opened, window)

    assert tuple(whole["label"].sizes[axis] for axis in ("y", "x")) == tuple(grid.shape)
    assert whole["s2"].sizes["time"] == 5


def test_instants_and_pixels_can_be_read_apart(items, opened) -> None:
    windows = cuts.chips(cuts.frames(cuts.stacks(items), 2, tolerance="10D"), 32)
    window = windows.iloc[0]

    framed = cuts.select_times(opened, window)
    chip = cuts.select_pixels(framed, window)

    assert framed["s2"].sizes == {"time": 2, "y": 50, "x": 70}
    assert dict(chip["s2"].sizes) == {"time": 2, "y": 32, "x": 32}


def test_unreferenced_pixels_are_cut_and_read_without_inventing_a_crs() -> None:
    source = xr.Dataset({"label": (("y", "x"), np.arange(12).reshape(3, 4))})

    windows = cuts.chips(whole_windows({"scene": source}), 2, mode="edge")
    row = windows.set_index("id").loc["scene/chip-3"]
    tile = cuts.select_pixels(source, row.to_dict())

    np.testing.assert_array_equal(tile.label.values, [[10, 11], [10, 11]])
    assert row["crs"] is None and row.geometry is None
    assert tile.gs.geobox is None


@pytest.mark.parametrize("kind", ["rotated", "wkt"])
def test_a_chip_states_its_exact_grid_on_any_grid(kind) -> None:
    crs = (
        "EPSG:32748"
        if kind == "rotated"
        else "+proj=aeqd +lat_0=17.123 +lon_0=42.456 +datum=WGS84 +units=m +no_defs"
    )
    transform = (
        Affine.translation(100, 500) * Affine.rotation(30) * Affine.scale(10, -10)
        if kind == "rotated"
        else Affine(10, 0, 100, 0, -10, 500)
    )
    grid = GeoBox((12, 16), transform, crs)
    parent = xr.Dataset({"B": (("y", "x"), np.ones((12, 16)))}, coords=xr_coords(grid))

    windows = cuts.chips(whole_windows({"scene": parent}), (6, 8), overlap=2)

    for row in windows.itertuples():
        stated = GeoBox((row.height, row.width), Affine(*row.transform), row.crs)
        assert stated == grid.translate_pix(row.col_off, row.row_off).crop(
            (row.height, row.width)
        )


def test_persisted_windows_keep_what_a_model_reads_from_a_row(tmp_path) -> None:
    import torch

    from geosave_engine.model.encoder import prithvi

    grid = GeoBox.from_bbox((10, 50, 12, 52), "EPSG:4326", shape=(6, 8))
    image = raster(
        {"red": (("y", "x"), np.arange(48, dtype="float32").reshape(6, 8))}, grid
    ).expand_dims(time=np.array(["2024-12-31", "2024-01-01"], dtype="datetime64[D]"))
    windows = cuts.chips(whole_windows({"scene": image}), 2)

    path = geoparquet.write(windows, tmp_path / "windows.parquet", index=False)
    restored = geoparquet.read(path).sample(frac=1, random_state=4)
    row = restored.set_index("id", drop=False).loc["scene/chip-11"]

    context = prithvi.model_context(row)
    torch.testing.assert_close(
        context["temporal_coords"], torch.tensor([[2024.0, 365.0], [2024.0, 0.0]])
    )
    tile = cuts.select_pixels(image, row.to_dict())
    np.testing.assert_array_equal(tile.red.isel(time=0).values, [[38, 39], [46, 47]])
