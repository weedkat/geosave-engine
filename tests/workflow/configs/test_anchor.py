import json

import numpy as np
from pydantic import TypeAdapter

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.configs import AnchorConfig


def test_coordinate_anchor_opens_a_native_grid():
    anchor = TypeAdapter(AnchorConfig).validate_python(
        {
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": [4, 6],
            "resolution": 10,
            "crs": "EPSG:32633",
            "timespan": "2025-01",
        }
    ).open()

    assert anchor.geobox.shape.yx == (4, 6)
    assert anchor.timespan is not None


def test_geojson_anchor_opens_a_native_grid(tmp_path):
    path = tmp_path / "area.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [[12, 45], [12.01, 45], [12.01, 45.01], [12, 45.01], [12, 45]]
                ],
            }
        )
    )

    anchor = TypeAdapter(AnchorConfig).validate_python(
        {"kind": "geojson", "path": path, "shape": [4, 4]}
    ).open()

    assert anchor.geobox.shape.yx == (4, 4)


def test_raster_anchor_uses_the_file_grid_and_time(tmp_path, anchor):
    instant = np.datetime64("2025-01-15")
    label = raster(
        {"class": np.ones((4, 4), dtype="uint8")}, anchor.geobox
    ).assign_coords(time=instant)
    path = io.geotiff.write_cog(label, tmp_path / "label.tif")

    result = TypeAdapter(AnchorConfig).validate_python(
        {"kind": "raster", "path": path}
    ).open()

    assert result.geobox == label.gs.anchor.geobox
    assert result.timespan == label.gs.anchor.timespan
