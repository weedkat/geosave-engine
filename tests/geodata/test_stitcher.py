from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs import Packing
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.core.stitcher import GeoStitcher

UTM = "EPSG:32633"
SHAPE = (16, 16)
OVERLAP = 4


def geobox(shape: tuple[int, int] = (32, 32)) -> GeoBox:
    """Build one grid for a stitching test.

    Args:
        shape: Grid height and width in pixels.

    Returns:
        Grid of `shape` at ten-metre pixels.
    """
    left, bottom = 300_000.0, 5_000_000.0
    return GeoBox.from_bbox(
        (left, bottom, left + shape[1] * 10, bottom + shape[0] * 10),
        crs=UTM,
        shape=shape,
        tight=True,
    )


def build(*, nodata: float | int | None = None, **extra: np.ndarray) -> xr.Dataset:
    """Build one raster whose pixels are all distinct.

    Args:
        nodata: Fill value every variable declares, or None for no declaration.
        **extra: Further variables to carry alongside `red`.

    Returns:
        Placed raster holding `red` and every name in `extra`.
    """
    box = geobox()
    values = np.arange(box.shape[0] * box.shape[1], dtype="float32").reshape(box.shape)
    built = raster({"red": values, **extra}, box)
    return built if nodata is None else built.gs.write_nodata(nodata)


def cut(source: xr.Dataset, *, group_id: str = "s") -> list[xr.Dataset]:
    """Cut a source into the tiles a stitching test lays back.

    Args:
        source: Raster to cut.
        group_id: Identifier the tiles share.

    Returns:
        Every tile of one tiling operation.
    """
    return source.gs.tile(SHAPE, group_id=group_id, overlap=OVERLAP)


def test_stitcher_rebuilds_the_source_exactly() -> None:
    source = build()
    stitcher = GeoStitcher()

    stitcher.add(*cut(source))
    (rebuilt,) = stitcher.flush()

    assert np.array_equal(rebuilt.red.values, source.red.values)
    assert rebuilt.gs.geobox == source.gs.geobox
    assert rebuilt.y.attrs["standard_name"] == "projection_y_coordinate"


def test_stitcher_does_not_follow_arrival_order() -> None:
    source = build()
    tiles = cut(source)
    stitcher = GeoStitcher()

    stitcher.add(*reversed(tiles))
    (rebuilt,) = stitcher.flush()

    assert np.array_equal(rebuilt.red.values, source.red.values)


def test_stitcher_releases_a_group_it_rebuilds() -> None:
    stitcher = GeoStitcher()
    stitcher.add(*cut(build()))

    stitcher.flush()

    assert len(stitcher) == 0
    assert stitcher.group_ids == ()


def test_stitcher_routes_a_mixed_batch_by_group() -> None:
    first, second = build(), build(red=np.ones((32, 32), "float32"))
    tiles = cut(first, group_id="A") + cut(second, group_id="B")
    stitcher = GeoStitcher()

    stitcher.add(*tiles)

    assert sorted(stitcher.group_ids) == ["A", "B"]
    assert len(stitcher.flush()) == 2


def test_stitcher_carries_the_attrs_without_the_tiling_record() -> None:
    source = build().assign_attrs(title="scene")
    stitcher = GeoStitcher()

    stitcher.add(*cut(source))
    (rebuilt,) = stitcher.flush()

    assert rebuilt.attrs["title"] == "scene"
    assert "tile_index" not in rebuilt.attrs
    assert "source_id" not in rebuilt.attrs


def test_stitcher_rebuilds_several_variables_over_a_leading_dim() -> None:
    box = geobox()
    values = np.random.default_rng(0).random((3, *box.shape)).astype("float32")
    source = raster(
        {"probability": values, "logit": values * 2}, box, **{"class": [0, 1, 2]}
    )
    stitcher = GeoStitcher()

    stitcher.add(*source.gs.tile(SHAPE, group_id="s", overlap=OVERLAP))
    (rebuilt,) = stitcher.flush()

    assert rebuilt.probability.dims == ("class", *box.dimensions)
    assert np.allclose(rebuilt.probability.values, source.probability.values)
    assert np.allclose(rebuilt.logit.values, source.logit.values)


def test_stitcher_drains_only_the_groups_that_are_complete() -> None:
    whole, partial = build(), build()
    stitcher = GeoStitcher()

    stitcher.add(*cut(whole, group_id="A"))
    stitcher.add(*cut(partial, group_id="B")[:-1])
    drained = list(stitcher.drain())

    assert len(drained) == 1
    assert stitcher.group_ids == ("B",)


def test_stitcher_reports_the_tiles_a_group_still_needs() -> None:
    tiles = cut(build())
    stitcher = GeoStitcher()

    stitcher.add(*tiles[:-2])

    assert stitcher.missing("s") == (len(tiles) - 2, len(tiles) - 1)
    assert not stitcher.is_complete("s")
    assert stitcher.missing("absent") == ()


def test_stitcher_refuses_a_partial_group_by_default() -> None:
    stitcher = GeoStitcher()
    stitcher.add(*cut(build())[:-1])

    with pytest.raises(ValueError, match="allow_partial"):
        stitcher.flush()


def test_stitcher_marks_a_partial_group_with_its_fill() -> None:
    stitcher = GeoStitcher()
    stitcher.add(*cut(build(nodata=-999))[:-1])

    (rebuilt,) = stitcher.flush(allow_partial=True)

    assert float(rebuilt.red.values[-1, -1]) == -999.0


def test_stitcher_refuses_a_partial_group_that_states_no_fill() -> None:
    stitcher = GeoStitcher()
    stitcher.add(*cut(build())[:-1])

    with pytest.raises(ValueError, match="no fill value"):
        stitcher.flush(allow_partial=True)


def test_stitcher_refuses_the_same_tile_twice() -> None:
    tiles = cut(build())
    stitcher = GeoStitcher()
    stitcher.add(tiles[0])

    with pytest.raises(ValueError, match="already added"):
        stitcher.add(tiles[0])


def test_stitcher_refuses_a_raster_that_states_no_tiling() -> None:
    with pytest.raises(ValueError, match="states no tiling"):
        GeoStitcher().add(build())


def test_stitcher_refuses_a_window_for_tiles_cut_without_overlap() -> None:
    tiles = build().gs.tile(SHAPE, group_id="s")

    with pytest.raises(ValueError, match="needs tiles cut"):
        GeoStitcher(window="hann").add(*tiles)


def test_stitcher_refuses_a_window_tiler_does_not_know() -> None:
    with pytest.raises(ValueError, match="not a window"):
        GeoStitcher(window="hanning")


def test_stitcher_refuses_variables_on_different_dtypes() -> None:
    source = build(nir=np.ones((32, 32), "uint8"))

    with pytest.raises(ValueError, match="different dtypes"):
        GeoStitcher().add(*cut(source))


def test_stitcher_refuses_variables_declaring_different_fills() -> None:
    source = build(nir=np.ones((32, 32), "float32"))
    source = source.gs.rebase(Packing(fill_value=-1), target="red")
    source = source.gs.rebase(Packing(fill_value=-2), target="nir")

    with pytest.raises(ValueError, match="different fill values"):
        GeoStitcher().add(*cut(source))


def test_stitcher_rebuilds_a_source_declaring_a_nan_fill() -> None:
    source = build(nodata=np.nan)
    stitcher = GeoStitcher()

    stitcher.add(*cut(source))
    (rebuilt,) = stitcher.flush()

    assert np.array_equal(rebuilt.red.values, source.red.values, equal_nan=True)


def test_stitcher_refuses_tiles_carrying_different_leading_coords() -> None:
    box = geobox()
    values = np.zeros((2, *box.shape), "float32")
    early = raster({"red": values}, box, time=pd.date_range("2024-01-01", periods=2))
    late = raster({"red": values}, box, time=pd.date_range("2025-06-01", periods=2))
    stitcher = GeoStitcher()

    stitcher.add(early.gs.tile(SHAPE, group_id="s", overlap=OVERLAP)[0])
    with pytest.raises(ValueError, match="different \\['time'\\] coordinates"):
        stitcher.add(late.gs.tile(SHAPE, group_id="s", overlap=OVERLAP)[1])


def test_stitcher_marks_the_frame_a_tapering_window_leaves_unweighted() -> None:
    source = build(nodata=-999)
    stitcher = GeoStitcher(window="hann")

    stitcher.add(*cut(source))
    (rebuilt,) = stitcher.flush()

    assert float(rebuilt.red.values[0, 0]) == -999.0
    assert np.allclose(
        rebuilt.red.values[8:24, 8:24], source.red.values[8:24, 8:24], atol=1e-2
    )


def test_stitcher_refuses_a_tapering_window_with_no_fill_to_mark_the_frame() -> None:
    stitcher = GeoStitcher(window="hann")
    stitcher.add(*cut(build()))

    with pytest.raises(ValueError, match="unweighted"):
        stitcher.flush()
