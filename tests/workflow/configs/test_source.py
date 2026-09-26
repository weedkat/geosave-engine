import math

from pydantic import ValidationError
import pytest

from geosave_engine.workflow.configs import QueryConfig, SourceConfig


def test_source_config_builds_native_query_and_load_settings():
    config = SourceConfig.model_validate(
        {
            "query": {
                "datetime": ["2025-01-01", "2025-01-03"],
                "max_items": 5,
                "sortby": [{"field": "properties.datetime", "direction": "desc"}],
            },
            "load": {"bands": ["B04", "B08"], "chunks": {"x": 32, "y": 16}},
        }
    )

    assert config.query.to_query("optical").to_search_params() == {
        "collections": ["optical"],
        "datetime": ("2025-01-01", "2025-01-03"),
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "max_items": 5,
    }
    assert config.load.bands == ["B04", "B08"]
    assert config.load.chunks == {"x": 32, "y": 16}


def test_source_config_names_a_prefect_concurrency_limit():
    config = SourceConfig.model_validate({"concurrency": "cdse"})

    assert config.concurrency == "cdse"


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": True},
        {"query": {"unexpected": True}},
        {"load": {"unexpected": True}},
        {"query": {"bbox": [-181, -10, 20, 10]}},
        {"query": {"max_items": 0}},
        {"query": {"limit": math.inf}},
        {"query": {"filter": {"value": math.nan}}},
        {"load": {"nodata": math.nan}},
        {"load": {"patch_url": lambda url: url}},
    ],
)
def test_invalid_source_settings_fail_during_config_validation(payload):
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(payload)


def test_query_rejects_empty_item_ids():
    with pytest.raises(ValidationError, match="ids"):
        QueryConfig(ids=())
