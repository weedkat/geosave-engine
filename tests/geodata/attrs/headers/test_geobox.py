from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs.headers import geobox


def test_geobox_factory_describes_its_spatial_coordinates() -> None:
    grid = GeoBox.from_bbox(
        (300_000, 5_000_000, 300_020, 5_000_020),
        crs="EPSG:32633",
        resolution=10,
    )

    header = geobox.create_header(grid)

    assert header.coords["y"].to_attrs() == {
        "standard_name": "projection_y_coordinate",
        "units": "metre",
        "axis": "Y",
    }
    assert header.coords["x"].to_attrs() == {
        "standard_name": "projection_x_coordinate",
        "units": "metre",
        "axis": "X",
    }
