import math

from pydantic import ValidationError
import pytest

from geosave_engine.model.spec import RasterRequirement, StacRecipe


def test_stac_recipe_preserves_priority_and_native_settings():
    recipe = StacRecipe.model_validate(
        {
            "collection": "sentinel-2-l2a",
            "endpoints": ["https://primary.test/stac", "http://backup.test/stac"],
            "query": {
                "datetime": ["2025-01-01", "2025-01-03"],
                "max_items": 5,
                "sortby": [
                    {"field": "properties.datetime", "direction": "desc"}
                ],
            },
            "load": {"bands": ["B04", "B08"], "chunks": {"x": 32, "y": 16}},
        }
    )

    assert recipe.model_dump()["endpoints"] == (
        "https://primary.test/stac",
        "http://backup.test/stac",
    )
    assert recipe.model_dump(mode="json")["endpoints"] == [
        "https://primary.test/stac",
        "http://backup.test/stac",
    ]
    assert recipe.query.to_query(recipe.collection).to_search_params() == {
        "collections": ["sentinel-2-l2a"],
        "datetime": ("2025-01-01", "2025-01-03"),
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "max_items": 5,
    }
    assert recipe.load.bands == ["B04", "B08"]
    assert StacRecipe.model_validate(recipe.model_dump()) == recipe


def test_stac_recipe_accepts_named_providers_with_url_fallbacks():
    recipe = StacRecipe.model_validate(
        {
            "collection": "sentinel-2-l2a",
            "endpoints": [
                "planetary_computer",
                "cdse",
                "element84",
                "https://custom.test/stac",
            ],
        }
    )

    assert recipe.model_dump()["endpoints"] == (
        "planetary_computer",
        "cdse",
        "element84",
        "https://custom.test/stac",
    )


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"collection": "", "endpoints": ["https://primary.test/stac"]},
        {"collection": "sentinel-2-l2a", "endpoints": []},
        {"collection": "sentinel-2-l2a", "endpoints": ["file:///tmp/catalog"]},
        {"collection": "sentinel-2-l2a", "endpoints": ["not a URL"]},
        {
            "collection": "sentinel-2-l2a",
            "endpoints": ["https://primary.test/stac", "https://primary.test/stac"],
        },
    ],
)
def test_stac_recipe_requires_collection_and_unique_http_endpoints(settings):
    with pytest.raises(ValidationError):
        StacRecipe.model_validate(settings)


def test_query_rejects_bbox_with_intersects():
    with pytest.raises(ValidationError, match="bbox.*intersects"):
        StacRecipe.model_validate(
            {
                "collection": "sentinel-2-l2a",
                "endpoints": ["https://primary.test/stac"],
                "query": {
                    "bbox": [0, 0, 1, 1],
                    "intersects": {"type": "Point", "coordinates": [0, 0]},
                },
            }
        )


@pytest.mark.parametrize(
    "load",
    [
        {"nodata": math.nan},
        {"patch_url": lambda url: url},
    ],
)
def test_stac_load_rejects_non_primitive_values(load):
    with pytest.raises(ValidationError):
        StacRecipe.model_validate(
            {
                "collection": "sentinel-2-l2a",
                "endpoints": ("https://primary.test/stac",),
                "load": load,
            }
        )


def test_named_variables_must_match_explicit_stac_bands():
    recipe = {
        "collection": "sentinel-2-l2a",
        "endpoints": ["https://primary.test/stac"],
        "load": {"bands": ["B08", "B04"]},
    }

    with pytest.raises(ValidationError, match="bands.*variables"):
        RasterRequirement.model_validate(
            {"variables": ("B04", "B08"), "stac": recipe}
        )

    requirement = RasterRequirement.model_validate(
        {"variables": ("B08", "B04"), "stac": recipe}
    )
    assert requirement.stac is not None


def test_positional_channels_accept_explicit_stac_bands():
    requirement = RasterRequirement.model_validate(
        {
            "channels": 2,
            "stac": {
                "collection": "sentinel-2-l2a",
                "endpoints": ["https://primary.test/stac"],
                "load": {"bands": ["B04", "B08"]},
            },
        }
    )

    assert requirement.stac is not None
    assert requirement.stac.load.bands == ["B04", "B08"]
