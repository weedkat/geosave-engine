import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from odc.geo.geobox import GeoBox
from tiler import Tiler

from geosave_engine.geodata import raster
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.transform.chip import chip_windows, crop
from tests.geodata.transform.test_vector import _band


@pytest.mark.parametrize("kind", ["array", "dataset", "tree"])
def test_pixel_window_preserves_native_type_time_and_grid_lazily(kind):
    band = _band(times=2).chunk().assign_attrs(description="reflectance")
    source = band if kind == "array" else band.to_dataset()
    if kind == "tree":
        source = stack({"image": source})

    with Callback(pretask=lambda *_: pytest.fail("window computed pixels")):
        result = crop(source, (2, 2), (3, 3), mode="edge")

    assert type(result) is type(source)
    if kind == "tree":
        output = result.gs.rasters["image"].red
    elif kind == "dataset":
        output = result.red
    else:
        output = result
    assert output.shape == (2, 3, 3)
    assert isinstance(output.data, da.Array)
    assert output.attrs["description"] == "reflectance"
    np.testing.assert_array_equal(output.time, band.time)
    assert output.gs.geobox == band.gs.geobox.translate_pix(2, 2).crop((3, 3))
    np.testing.assert_array_equal(output.values, np.ones((2, 3, 3)))


@pytest.mark.parametrize("halo", [False, True])
@pytest.mark.parametrize("kind", ["rotated", "wkt", "plain", "mixed"])
def test_tile_reference_keeps_native_identity_and_exact_grid(kind, halo):
    import dask.array as da
    from affine import Affine
    from geopandas.testing import assert_geodataframe_equal
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords
    from tiler import Tiler

    crs = (
        "EPSG:32748"
        if kind != "wkt"
        else "+proj=aeqd +lat_0=17.123 +lon_0=42.456 +datum=WGS84 +units=m +no_defs"
    )
    grid = GeoBox((13, 19), Affine(10, 2, 600000, 1, -10, 9000000), crs)
    geo = xr.DataArray(
        da.ones((13, 19), chunks=(4, 5)), dims=("y", "x"), coords=xr_coords(grid)
    )
    plain = xr.DataArray(da.ones((13, 19), chunks=(4, 5)), dims=("y", "x"))
    parents = {
        "a": plain if kind == "plain" else geo,
        "b": plain if kind in ("plain", "mixed") else geo,
    }
    tilers, padding = {}, {}
    for key, parent in parents.items():
        tiler = Tiler(parent.shape, (6, 8), overlap=2, mode="reflect")
        padding[key] = [(0, 0), (0, 0)]
        if halo:
            shape, padding[key] = tiler.calculate_padding()
            tiler.recalculate(data_shape=shape)
        tilers[key] = tiler
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        reference = chip_windows(parents, tilers, padding=padding)
    assert tasks == []
    assert reference.id.is_unique
    assert len(reference) == sum(map(len, tilers.values()))
    for _, row in reference.iterrows():
        near, far = tilers[row.parent_id].get_tile_bbox(int(row.tile_id))
        expected_offset = tuple(
            int(start) - int(width[0])
            for start, width in zip(near, padding[row.parent_id], strict=True)
        )
        assert (row.row_off, row.col_off) == expected_offset
        assert (row.height, row.width) == tuple(far - near)
        parent_grid = parents[row.parent_id].odc.geobox
        if parent_grid is None:
            assert row.geometry is None
            assert all(
                row[name] is None
                for name in ("proj:shape", "proj:transform", "proj:code", "proj:wkt2")
            )
        else:
            code = row["proj:code"] or row["proj:wkt2"]
            recovered = GeoBox(row["proj:shape"], Affine(*row["proj:transform"]), code)
            assert recovered == parent_grid.translate_pix(
                row.col_off, row.row_off
            ).crop((row.height, row.width))
    reordered = chip_windows(
        dict(reversed(list(parents.items()))), tilers, padding=padding
    )
    assert_geodataframe_equal(
        reference.set_index("id").sort_index(), reordered.set_index("id").sort_index()
    )
    assert (reference.crs is None) == (kind == "plain")


def test_tile_reference_requires_matching_parent_keys():
    from tiler import Tiler

    source = xr.DataArray(np.ones((4, 4)), dims=("y", "x"))
    with pytest.raises(ValueError, match="parent"):
        chip_windows({"a": source}, {"b": Tiler((4, 4), (2, 2))})


def _window_raster(height=6, width=8):
    grid = GeoBox.from_bbox((10, 50, 12, 52), "EPSG:4326", shape=(height, width))
    return raster(
        {"red": np.arange(height * width, dtype="float32").reshape(height, width)}, grid
    )


def test_filtered_reference_reads_by_id_without_loading_other_tiles():
    import dask.array as da
    from dask.callbacks import Callback

    source = _window_raster().chunk({"latitude": 2, "longitude": 2})
    parents = {"scene": source}
    tilers = {"scene": Tiler((6, 8), (2, 2))}
    with Callback(
        pretask=lambda *_: (_ for _ in ()).throw(AssertionError("read pixels at setup"))
    ):
        reference = chip_windows(parents, tilers)
        reference = reference.iloc[[11, 0]]
    tile = reference.set_index("id").loc["scene/tile-11"].gs.crop(source)
    np.testing.assert_array_equal(tile.red.values, [[38, 39], [46, 47]])
    assert isinstance(tile.red.data, da.Array)
    assert tile.gs.geobox == _window_raster().gs.geobox.translate_pix(6, 4).crop((2, 2))


def test_reference_keeps_each_rasters_timestamp_order_without_pixels():
    from dask.callbacks import Callback

    image = _window_raster().expand_dims(
        time=np.array(["2024-12-31T14:00", "2024-01-01T03:00"], dtype="datetime64[m]")
    )
    parent = stack({"image": image.chunk(), "label": _window_raster().chunk()})
    with Callback(
        pretask=lambda *_: (_ for _ in ()).throw(AssertionError("read pixels at setup"))
    ):
        reference = chip_windows({"scene": parent}, {"scene": Tiler((6, 8), (2, 2))})
    metadata = reference.iloc[0].raster_metadata
    assert metadata["image"]["times"] == ["2024-12-31T14:00:00", "2024-01-01T03:00:00"]
    assert metadata["image"]["bands"] == ["red"]
    assert metadata["label"]["times"] == []


def test_indexed_merger_reassembles_shuffled_predictions():
    from tiler import Merger

    parent = _window_raster()
    parents = {"scene": parent}
    tilers = {"scene": Tiler((6, 8), (4, 4), overlap=2)}
    reference = chip_windows(parents, tilers)
    reference = reference.set_index("id", drop=False)
    merger = Merger(tilers["scene"], logits=1, save_visits=False)
    for sample_id in reversed(reference.id.tolist()):
        row = reference.loc[sample_id]
        tile = row.gs.crop(parent)
        merger.add(int(row.tile_id), tile.red.values[None] * 2)
    np.testing.assert_allclose(merger.merge(), parent.red.values[None] * 2)


def test_plain_raster_rows_read_windows_without_inventing_a_crs():
    import xarray as xr

    source = xr.Dataset({"label": (("y", "x"), np.arange(12).reshape(3, 4))})
    reference = chip_windows(
        {"scene": source}, {"scene": Tiler((3, 4), (2, 2), mode="edge")}
    )
    tile = reference.set_index("id").loc["scene/tile-3"].gs.crop(source)
    np.testing.assert_array_equal(tile.label.values, [[10, 11], [10, 11]])
    assert tile.gs.geobox is None


def test_persisted_metadata_and_window_keep_row_context(tmp_path):
    from geosave_engine.geodata.io import geoparquet
    from geosave_engine.model.encoder import prithvi
    import torch

    image = _window_raster().expand_dims(
        time=np.array(["2024-12-31", "2024-01-01"], dtype="datetime64[D]")
    )
    reference = chip_windows({"scene": image}, {"scene": Tiler((6, 8), (2, 2))})
    path = geoparquet.write(reference, tmp_path / "tiles.parquet", index=False)
    restored = geoparquet.read(path).sample(frac=1, random_state=4)
    row = restored.set_index("id", drop=False).loc["scene/tile-11"]
    context = prithvi.model_context(row)
    torch.testing.assert_close(
        context["temporal_coords"], torch.tensor([[2024.0, 365.0], [2024.0, 0.0]])
    )
    tile = row.gs.crop(image)
    np.testing.assert_array_equal(tile.red.isel(time=0).values, [[38, 39], [46, 47]])


@pytest.mark.parametrize("mode", ["reflect", "wrap"])
@pytest.mark.parametrize("halo", [False, True])
@pytest.mark.parametrize("georeferenced", [False, True])
def test_large_lazy_padding_matches_native_tiler(mode, halo, georeferenced):
    import xarray as xr
    from dask.callbacks import Callback

    source = (
        _window_raster(3, 4)
        if georeferenced
        else xr.Dataset({"red": (("y", "x"), np.arange(12).reshape(3, 4))})
    )
    original = source.red.values
    tiler = Tiler((3, 4), (16, 16), overlap=2, mode=mode)
    padding = [(0, 0), (0, 0)]
    if halo:
        shape, padding = tiler.calculate_padding()
        tiler.recalculate(data_shape=shape)
    # Native Tiler may produce no tiles without a halo for a very small scene;
    # use a larger source to exercise repeated fringe padding in that case.
    if not halo:
        source = (
            _window_raster(13, 19)
            if georeferenced
            else xr.Dataset({"red": (("y", "x"), np.arange(247).reshape(13, 19))})
        )
        original = source.red.values
        tiler = Tiler((13, 19), (16, 16), overlap=2, mode=mode)
    reference = chip_windows(
        {"scene": source}, {"scene": tiler}, padding={"scene": padding}
    )
    padded = np.pad(original, padding, mode=mode)
    for row in reference.itertuples():
        with Callback(pretask=lambda *_: pytest.fail("window read computed pixels")):
            tile = reference.set_index("id").loc[row.id].gs.crop(source.chunk())
        assert tile.red.shape == (16, 16)
        np.testing.assert_array_equal(
            tile.red.values, tiler.get_tile(padded, row.tile_id)
        )

