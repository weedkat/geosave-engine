from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs.headers import geobox


def test_geobox_factory_describes_its_spatial_coordinates_completely() -> None:
    grid = GeoBox.from_bbox(
        (300_000, 5_000_000, 300_020, 5_000_020),
        crs="EPSG:32633",
        resolution=10,
    )

    header = geobox.create_header(grid)

    assert header.root.to_attrs() == {}
    assert header.coords["y"].to_attrs() == {
        "crs": "EPSG:32633",
        "resolution": -10.0,
        "standard_name": "projection_y_coordinate",
        "units": "metre",
        "axis": "Y",
    }
    assert header.coords["x"].to_attrs() == {
        "crs": "EPSG:32633",
        "resolution": 10.0,
        "standard_name": "projection_x_coordinate",
        "units": "metre",
        "axis": "X",
    }


def test_geographic_header_describes_y_x_coordinates() -> None:
    grid = GeoBox.from_bbox((10, 20, 13, 22), crs="EPSG:4326", shape=(2, 3))

    header = geobox.create_header(grid)

    assert set(header.coords) == {"y", "x"}
    assert header.coords["y"].to_attrs()["standard_name"] == "latitude"
    assert header.coords["x"].to_attrs()["standard_name"] == "longitude"
