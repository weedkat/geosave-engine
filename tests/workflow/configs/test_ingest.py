from pydantic import ValidationError
import pytest

from geosave_engine.workflow.configs import CoordinateAnchorConfig, IngestConfig


def settings(tmp_path):
    return {
        "sources": {"optical": {"query": {}, "load": {}}},
        "anchor": {
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": [4, 6],
            "resolution": 10,
        },
        "output": str(tmp_path / "raw.zarr"),
        "spec": str(tmp_path / "model_spec.yaml"),
    }


def test_ingest_config_dispatches_coordinate_anchor(tmp_path):
    config = IngestConfig.model_validate(settings(tmp_path))

    assert isinstance(config.anchor, CoordinateAnchorConfig)
    assert config.anchor.shape == (4, 6)


def test_unknown_nested_source_fields_fail_before_http(tmp_path):
    payload = settings(tmp_path)
    payload["sources"] = {"optical": {"query": {"unexpected": True}}}

    with pytest.raises(ValidationError, match="unexpected"):
        IngestConfig.model_validate(payload)


def test_source_names_are_checked_without_opening_any_source(tmp_path):
    payload = settings(tmp_path)
    payload["sources"] = {"extra": {}}
    config = IngestConfig.model_validate(payload)

    with pytest.raises(ValueError, match="missing.*optical"):
        config.validate_sources({"optical": object()})


@pytest.mark.parametrize(
    ("requirements", "message"),
    [
        ({"optical": object(), "terrain": object()}, "missing.*terrain"),
        ({"terrain": object()}, "missing.*terrain"),
        ({}, "At least one"),
    ],
)
def test_model_source_names_must_match_exactly(tmp_path, requirements, message):
    config = IngestConfig.model_validate(settings(tmp_path))

    with pytest.raises(ValueError, match=message):
        config.validate_sources(requirements)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output", "s3://bucket/raw.zarr"),
        ("output", "raw.nc"),
        ("spec", "https://example.test/model_spec.yaml"),
        ("spec", "model_spec.json"),
    ],
)
def test_paths_must_be_local_and_use_supported_suffixes(tmp_path, field, value):
    payload = settings(tmp_path)
    payload[field] = value

    with pytest.raises(ValidationError):
        IngestConfig.model_validate(payload)


def test_output_must_not_exist(tmp_path):
    payload = settings(tmp_path)
    (tmp_path / "raw.zarr").mkdir()

    with pytest.raises(ValidationError, match="already exists"):
        IngestConfig.model_validate(payload)
