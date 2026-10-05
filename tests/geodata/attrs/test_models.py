import re

import pytest

from geosave_engine.geodata.attrs import MODELS, Nodata, TimeSpec, resolve_model
from geosave_engine.geodata.attrs.models import model_scope


def test_each_attr_key_has_one_owner_per_scope() -> None:
    for models in MODELS.values():
        keys = [key for model in models for key in model.attr_keys()]
        assert len(keys) == len(set(keys))


def test_model_names_are_unique_across_scopes() -> None:
    names = [model.NAME for models in MODELS.values() for model in models]
    assert len(names) == len(set(names))


def test_every_model_resolves_by_name_and_class() -> None:
    for models in MODELS.values():
        for model in models:
            assert resolve_model(model.NAME) is model
            assert resolve_model(model) is model


def test_a_model_outside_the_table_is_refused() -> None:
    class Calibration(Nodata):
        NAME = "calibration"

    with pytest.raises(TypeError, match="not a GeoSave attrs model"):
        resolve_model(Calibration)
    with pytest.raises(KeyError, match="calibration"):
        resolve_model("calibration")


def test_scopes_follow_cf_usage() -> None:
    assert model_scope(Nodata) == "variable"
    assert model_scope(TimeSpec) == "coordinate"


def test_one_field_reads_any_of_its_spellings() -> None:
    assert Nodata.from_attrs({"nodata": 0}) == Nodata(fill_value=0)
    assert Nodata.from_attrs({"units": "1"}) is None
    with pytest.raises(ValueError, match="they spell one Nodata.fill_value"):
        Nodata.from_attrs({"_FillValue": 0, "nodata": -9999})


def test_model_names_are_their_class_in_snake_case() -> None:
    for models in MODELS.values():
        for model in models:
            # GeoTIFF is one format name, not two words.
            name = model.__name__.replace("GeoTIFF", "Geotiff")
            words = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
            assert model.NAME == words.lower()


def test_stac_metadata_merges_into_every_item_once() -> None:
    from datetime import datetime

    from geosave_engine.geodata.attrs.models.stac import StacItem, StacMetadata

    def loaded(*ids: str) -> StacMetadata:
        return StacMetadata(
            stac_groupby="solar_day",
            stac_items=tuple(
                StacItem(id=name, datetime=datetime(2025, 6, 1)) for name in ids
            ),
        )

    merged, dropped = StacMetadata.merge([loaded("a", "b"), loaded("b", "c")])

    assert [item.id for item in merged.stac_items or ()] == ["a", "b", "c"]
    assert merged.stac_groupby == "solar_day"
    assert dropped == set()


def test_rasters_loaded_from_stac_merge_their_bands() -> None:
    from datetime import datetime

    from geosave_engine.geodata.attrs.models.stac import StacItem, StacMetadata
    from geosave_engine.geodata.transform.merge import merge_bands
    from tests.geodata.conftest import build_raster

    def loaded(name: str) -> StacMetadata:
        return StacMetadata(
            stac_items=(StacItem(id=name, datetime=datetime(2025, 6, 1)),)
        )

    raster = build_raster()
    optical = raster[["red"]].gs.rebase(loaded("optical-scene"))
    radar = raster[["nir"]].gs.rebase(loaded("radar-scene"))

    merged = merge_bands([optical, radar])

    items = merged.gs.attrs.root.get(StacMetadata).stac_items
    assert [item.id for item in items] == ["optical-scene", "radar-scene"]
