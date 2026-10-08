"""Tests for eager and lazy spatial kernels with halos."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import uniform_filter


@pytest.mark.parametrize("backend", ["numpy", "dask", "mixed"])
def test_mapping_keeps_coordinates_and_matches_eager_neighborhoods(
    backend: str,
) -> None:
    from geosave_engine.geodata.utils.dask_mapping import map_spatial_overlap

    values = np.arange(64, dtype="float32").reshape(8, 8)
    scene = xr.Dataset(
        {"a": (("y", "x"), values), "b": (("y", "x"), values * 2)},
        coords={"y": np.arange(8), "x": np.arange(8)},
    )
    scene.a.attrs["units"] = "source units"
    a = scene.a.chunk({"y": 4, "x": 4}) if backend != "numpy" else scene.a
    b = scene.b.chunk({"y": 2, "x": 2}) if backend == "dask" else scene.b

    def smooth(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return uniform_filter(left + right, size=3)

    result = map_spatial_overlap(smooth, a, b, depth=1, dtype="float32")

    assert isinstance(result.data, da.Array) == (backend != "numpy")
    assert result.name is None
    assert result.attrs == {}
    assert result.dtype == np.float32
    xr.testing.assert_identical(result.coords.to_dataset(), scene.coords.to_dataset())
    np.testing.assert_allclose(result.compute(), uniform_filter(values * 3, size=3))


def test_mapping_never_overlaps_the_time_axis() -> None:
    from geosave_engine.geodata.utils.dask_mapping import map_spatial_overlap

    values = np.stack([np.ones((4, 4)), np.full((4, 4), 10)]).astype("float32")
    field = xr.DataArray(values, dims=("time", "y", "x")).chunk(
        {"time": 1, "y": 2, "x": 2}
    )

    # SciPy accepts per-axis sizes; its inferred signature only names int.
    result = map_spatial_overlap(
        lambda block: uniform_filter(block, size=(1, 3, 3)),  # pyright: ignore[reportArgumentType]
        field,
        depth=1,
        dtype="float32",
    )

    np.testing.assert_allclose(result.compute(), values)


def test_mapping_refuses_a_halo_that_needs_larger_chunks() -> None:
    from geosave_engine.geodata.utils.dask_mapping import map_spatial_overlap

    field = xr.DataArray(np.ones((8, 8)), dims=("y", "x")).chunk({"y": 2, "x": 2})

    with pytest.raises(ValueError):
        map_spatial_overlap(lambda block: block, field, depth=3, dtype="float32")


@pytest.mark.parametrize("lazy", [False, True])
def test_mapping_orders_dataset_bands_by_dimension_name(lazy: bool) -> None:
    from geosave_engine.geodata.utils.dask_mapping import map_spatial_overlap

    values = np.arange(16, dtype="float32").reshape(4, 4)
    scene = xr.Dataset({"a": (("y", "x"), values), "b": (("y", "x"), values * 2)})
    scene["b"] = scene.b.transpose("x", "y")
    if lazy:
        scene = scene.chunk({"y": 2, "x": 2})

    result = map_spatial_overlap(
        lambda a, b: a + b,
        scene.a,
        scene.b,
        depth=1,
        dtype="float32",
    )

    np.testing.assert_array_equal(result.compute(), values * 3)


@pytest.mark.parametrize("lazy", [False, True])
def test_mapping_refuses_negative_halos(lazy: bool) -> None:
    from geosave_engine.geodata.utils.dask_mapping import map_spatial_overlap

    field = xr.DataArray(np.ones((4, 4)), dims=("y", "x"))
    if lazy:
        field = field.chunk({"y": 2, "x": 2})

    with pytest.raises(ValueError):
        map_spatial_overlap(lambda block: block, field, depth=-1, dtype="float32")
