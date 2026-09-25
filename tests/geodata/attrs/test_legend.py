from __future__ import annotations

import pytest
from pydantic import ValidationError

from geosave_engine.geodata.attrs import Legend


def test_a_class_map_is_written_the_way_cf_spells_it() -> None:
    flags = Legend(class_map={1: "palm", 0: "bg"})

    assert flags.flag_values == [0, 1]
    assert flags.flag_meanings == "bg palm"
    assert flags.to_attrs() == {"flag_values": [0, 1], "flag_meanings": "bg palm"}


def test_the_class_map_reads_back_off_the_cf_pair() -> None:
    flags = Legend(flag_values=[0, 1], flag_meanings="bg palm")

    assert flags.class_map == {0: "bg", 1: "palm"}


def test_a_class_map_is_derived_rather_than_stored() -> None:
    # Nothing to sync: the listing is the one stored fact.
    assert "class_map" not in Legend.field_keys
    assert "class_map" not in Legend(class_map={0: "bg"}).to_attrs()


def test_a_bitfield_listing_enumerates_no_values() -> None:
    flags = Legend(flag_masks=[1, 2, 4], flag_meanings="cloud shadow snow")

    assert flags.class_map is None
    assert flags.flag_masks == [1, 2, 4]


def test_a_class_map_rejects_names_with_whitespace() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        Legend(class_map={0: "palm oil"})


def test_codes_and_names_must_match_in_number() -> None:
    with pytest.raises(ValidationError, match="names 3 classes"):
        Legend(flag_values=[0, 1], flag_meanings="bg palm shrub")


def test_codes_and_names_are_set_together() -> None:
    with pytest.raises(ValidationError, match="flag_meanings names no classes"):
        Legend(flag_values=[0, 1])

    with pytest.raises(ValidationError, match="no flag_values or flag_masks"):
        Legend(flag_meanings="bg palm")


def test_flag_values_must_be_ascending_and_unique() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        Legend(flag_values=[2, 0, 1], flag_meanings="a b c")


def test_codes_read_back_from_the_text_a_gdal_tag_holds() -> None:
    flags = Legend(flag_values="[0, 1]", flag_meanings="bg palm")

    assert flags.flag_values == [0, 1]


def test_a_legend_carries_colour_without_a_listing() -> None:
    legend = Legend(color_map={0: "#000000", 1: "#00ff00"})

    # to_attrs is JSON-native, so a mapping's keys come out as strings.
    assert legend.to_attrs() == {"color_map": {"0": "#000000", "1": "#00ff00"}}


def test_a_colour_keyed_to_no_listed_class_is_refused() -> None:
    with pytest.raises(ValidationError, match="does not name"):
        Legend(class_map={0: "bg", 1: "palm"}, color_map={0: "#000000", 5: "#00ff00"})


def test_colours_may_cover_fewer_classes_than_the_listing_names() -> None:
    legend = Legend(class_map={0: "bg", 1: "palm"}, color_map={0: "#000000"})

    assert legend.class_map == {0: "bg", 1: "palm"}


def test_assigning_a_class_map_rewrites_both_cf_fields() -> None:
    flags = Legend(class_map={0: "bg", 1: "palm"})

    flags.class_map = {0: "bg", 1: "oil", 2: "water"}

    assert flags.flag_values == [0, 1, 2]
    assert flags.flag_meanings == "bg oil water"


def test_editing_a_cf_field_shows_in_the_class_map() -> None:
    flags = Legend(class_map={0: "bg", 1: "palm"})

    flags.flag_meanings = "sea land"

    # One stored fact, so the view follows without anything keeping it in step.
    assert flags.class_map == {0: "sea", 1: "land"}
