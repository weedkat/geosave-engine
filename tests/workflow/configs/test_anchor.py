import json

from pydantic import TypeAdapter, ValidationError
import pytest

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.workflow.configs import (
    AnchorConfig,
    CoordinateAnchorConfig,
    GeoJSONAnchorConfig,
)


def test_coordinate_anchor_normalizes_lists_and_opens_native_anchor():
    config = TypeAdapter(AnchorConfig).validate_python(
        {
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": [4, 6],
            "resolution": 10,
            "crs": "EPSG:32633",
            "timespan": ["2025-01-01", "2025-01-03"],
        }
    )

    assert isinstance(config, CoordinateAnchorConfig)
    assert config.shape == (4, 6)
    assert config.timespan == ("2025-01-01", "2025-01-03")
    assert (
        config.open().geobox
        == GeoAnchor.from_coordinates(
            45,
            12,
            (4, 6),
            10,
            crs="EPSG:32633",
            timespan=("2025-01-01", "2025-01-03"),
        ).geobox
    )


def test_geojson_anchor_opens_the_file_only_when_requested(tmp_path):
    path = tmp_path / "area.geojson"
    config = TypeAdapter(AnchorConfig).validate_python(
        {"kind": "geojson", "path": path, "shape": [4, 6]}
    )
    assert isinstance(config, GeoJSONAnchorConfig)
    assert not path.exists()
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

    result = config.open()

    assert result.geobox.shape.yx == (4, 6)


@pytest.mark.parametrize("kind", [None, "point", "coordinates "])
def test_anchor_kind_is_required_and_exact(kind):
    payload = {
        "kind": kind,
        "latitude": 45,
        "longitude": 12,
        "shape": 4,
        "resolution": 10,
    }
    with pytest.raises(ValidationError, match="kind"):
        TypeAdapter(AnchorConfig).validate_python(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "coordinates",
            "latitude": 91,
            "longitude": 12,
            "shape": 4,
            "resolution": 10,
        },
        {
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": 0,
            "resolution": 10,
        },
        {
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": 4,
            "resolution": 0,
        },
        {"kind": "geojson", "path": "area.geojson"},
        {"kind": "geojson", "path": "area.geojson", "shape": 4, "resolution": 10},
        {"kind": "geojson", "path": "area.geojson", "shape": 4, "pad": -1},
        {"kind": "geojson", "path": "s3://bucket/area.geojson", "shape": 4},
        {"kind": "geojson", "path": "area.txt", "shape": 4},
    ],
)
def test_invalid_anchor_parameters_fail_during_config_validation(payload):
    with pytest.raises(ValidationError):
        TypeAdapter(AnchorConfig).validate_python(payload)
