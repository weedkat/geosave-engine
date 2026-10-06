from __future__ import annotations

import numpy as np
import pytest


def test_crop_cuts_a_band_and_marks_outside_pixels_nodata() -> None:
    import geopandas as gpd
    import shapely
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.array import array

    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    band = array(np.ones(grid.shape, "uint16"), grid, nodata=0)
    triangle = gpd.GeoDataFrame(
        geometry=[shapely.Polygon([(0, 0), (20, 0), (0, 20)])], crs="EPSG:32633"
    )

    cropped = band.gs.crop(triangle)

    assert cropped.shape == (2, 2)
    assert cropped.dtype == np.dtype("uint16")
    assert (cropped.values == 0).any() and (cropped.values == 1).any()


def test_crop_refuses_a_raster_without_crs_and_names_the_accessor() -> None:
    import geopandas as gpd
    import shapely
    import xarray as xr

    from geosave_engine.geodata.transform.vector import crop

    band = xr.DataArray(np.ones((4, 4), "uint16"), dims=("y", "x"))
    square = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 2, 2)], crs="EPSG:32633")

    with pytest.raises(ValueError, match=r"gs\.write_crs"):
        crop(band, square)


def _triangle():
    import geopandas as gpd
    import shapely

    return gpd.GeoDataFrame(
        geometry=[shapely.Polygon([(0, 0), (30, 0), (0, 30)])], crs="EPSG:32633"
    )


def _band(times: int = 0):
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.array import array

    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    band = array(np.ones(grid.shape, "uint16"), grid, nodata=0).rename("red")
    if not times:
        return band
    return band.expand_dims(time=np.arange(times).astype("datetime64[D]")).copy()


def test_crop_masks_pixel_centres_inside_the_geometry() -> None:
    cropped = _band().gs.crop(_triangle())

    assert cropped.values.tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_masks_every_instant_of_a_time_axis() -> None:
    cropped = _band(times=2).gs.crop(_triangle())

    assert cropped.shape == (2, 3, 3)
    assert cropped.values[0].tolist() == cropped.values[1].tolist()
    assert cropped.values[1].tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_cuts_every_group_of_a_stack_and_keeps_root_attrs() -> None:
    import geosave_engine.geodata.attrs as attrs
    from geosave_engine.geodata.core.stack import stack
    from geosave_engine.geodata.transform.vector import crop

    band = _band()
    tree = stack(
        {"optical": band.to_dataset(), "dem": band.rename("height").to_dataset()}
    ).gs.rebase(attrs.ACDD(title="scene"))

    cropped = crop(tree, _triangle())

    assert cropped.gs.groups == ("optical", "dem")
    assert cropped.gs.geobox.shape == (3, 3)
    assert cropped.gs.rasters["dem"].height.values.tolist() == [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
    assert cropped.gs.attrs.root.get(attrs.ACDD).title == "scene"


def test_crop_refuses_an_empty_vector() -> None:
    from geosave_engine.geodata.core.vector import GeoVector

    with pytest.raises(ValueError, match="empty vector"):
        _band().gs.crop(GeoVector.empty("EPSG:32633"))


def test_crop_masks_a_geographic_grid_whose_coordinates_are_not_float_exact() -> None:
    import geopandas as gpd
    import shapely
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.array import array

    step = 1 / 3600
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.0 + 20 * step, 45.0 + 20 * step),
        crs="EPSG:4326",
        resolution=step,
    )
    band = array(np.ones(grid.shape, "uint16"), grid, nodata=0)
    corner = gpd.GeoDataFrame(
        geometry=[
            shapely.Polygon(
                [
                    (13.0 + 3.3 * step, 45.0 + 3.3 * step),
                    (13.0 + 9.7 * step, 45.0 + 3.3 * step),
                    (13.0 + 3.3 * step, 45.0 + 9.7 * step),
                ]
            )
        ],
        crs="EPSG:4326",
    )

    cropped = band.gs.crop(corner)

    assert cropped.dtype == np.dtype("uint16")
    assert (cropped.values == 0).any() and (cropped.values == 1).any()


def test_crop_masks_a_raster_whose_leading_axis_holds_no_dates() -> None:
    band = _band().expand_dims(time=[0, 1]).copy()

    cropped = band.gs.crop(_triangle())

    assert cropped.shape == (2, 3, 3)
    assert cropped.values[1].tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_reads_no_pixels_of_a_chunked_raster() -> None:
    cropped = _band(times=2).chunk().gs.crop(_triangle())

    assert cropped.chunks is not None


def _sliced_geographic_scene():
    """Build a raster whose sliced coordinates drift from its geobox's own."""
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.raster import raster

    step = 1 / 3600
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.0 + 40 * step, 45.0 + 40 * step),
        crs="EPSG:4326",
        resolution=step,
    )
    whole = raster({"red": np.ones(grid.shape, "uint16")}, grid, nodata=0)
    return whole.isel(longitude=slice(5, 25), latitude=slice(5, 25))


def test_rasterize_lands_on_the_target_rasters_own_coordinates() -> None:
    import geopandas as gpd
    import shapely
    import xarray as xr

    from geosave_engine.geodata.transform.nodata import mask

    scene = _sliced_geographic_scene()
    left, bottom, right, top = scene.gs.bounds.bbox
    plot = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
        crs="EPSG:4326",
    )

    burned = plot.gs.rasterize(scene)

    xr.align(scene, burned, join="exact")
    masked = mask(scene, burned)
    assert (masked.red.values == 0).any() and (masked.red.values == 1).any()


def _flags(values, dtype):
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.array import array

    pixels = np.asarray(values, dtype=dtype)
    side = pixels.shape[0] * 10
    grid = GeoBox.from_bbox((0, 0, side, side), crs="EPSG:32633", resolution=10)
    return array(pixels, grid)


def test_vectorize_joins_regions_touching_at_a_corner_into_valid_geometry() -> None:
    from geosave_engine.geodata.core.vector import GeoVector

    diagonal = _flags([[1, 0], [0, 1]], "uint8")

    regions = GeoVector.vectorize(diagonal, connectivity=8)

    assert regions.geometry.is_valid.all()
    ones = regions[regions["value"] == 1]
    assert ones.geometry.area.sum() == 200


@pytest.mark.parametrize("dtype", ["int64", "int8", "uint32"])
def test_vectorize_takes_the_integer_dtypes_a_model_emits(dtype: str) -> None:
    from geosave_engine.geodata.core.vector import GeoVector

    regions = GeoVector.vectorize(_flags([[1, 0], [0, 1]], dtype))

    assert sorted(regions["value"].tolist()) == [0, 0, 1, 1]


def test_vectorize_refuses_integers_no_supported_dtype_holds() -> None:
    from geosave_engine.geodata.core.vector import GeoVector

    with pytest.raises(ValueError, match="int32"):
        GeoVector.vectorize(_flags([[2**40, 0], [0, 1]], "int64"))


def test_crop_takes_a_vector_in_another_crs() -> None:
    import geopandas as gpd
    import shapely

    # Corners off the pixel edges, so a round trip through degrees moves no edge.
    triangle = gpd.GeoDataFrame(
        geometry=[shapely.Polygon([(3, 3), (27, 3), (3, 27)])], crs="EPSG:32633"
    )

    native = _band().gs.crop(triangle)
    geographic = _band().gs.crop(triangle.to_crs("EPSG:4326"))

    assert geographic.shape == native.shape
    assert geographic.values.tolist() == native.values.tolist()


def test_crop_takes_an_odc_geometry_carrying_its_own_crs() -> None:
    from odc.geo.geom import box

    cropped = _band().gs.crop(box(10, 10, 30, 30, "EPSG:32633"))

    assert cropped.shape == (2, 2)
    assert cropped.values.tolist() == [[1, 1], [1, 1]]


def test_crop_takes_a_geoseries() -> None:
    cropped = _band().gs.crop(_triangle().geometry)

    assert cropped.values.tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_crop_keeps_every_disconnected_geometry_and_masks_the_gap() -> None:
    import geopandas as gpd
    import shapely

    corners = gpd.GeoDataFrame(
        geometry=[shapely.box(0, 0, 10, 10), shapely.box(30, 30, 40, 40)],
        crs="EPSG:32633",
    )

    cropped = _band().gs.crop(corners)

    assert cropped.shape == (4, 4)
    # North is the first row, so the box at y 30-40 is the top-right pixel.
    assert cropped.values.tolist() == [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]


def test_crop_refuses_a_bare_shapely_geometry_that_names_no_crs() -> None:
    import shapely

    with pytest.raises(TypeError, match="CRS"):
        _band().gs.crop(shapely.box(10, 10, 30, 30))


def test_crop_refuses_an_odc_geometry_that_names_no_crs() -> None:
    from odc.geo.geom import box

    with pytest.raises(TypeError, match="CRS"):
        _band().gs.crop(box(10, 10, 30, 30, None))
