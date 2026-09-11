from __future__ import annotations

import pytest
from pydantic import ValidationError

from geosave_engine.geodata.attrs import Legend


def test_class_map_derives_the_cf_flag_pair() -> None:
    legend = Legend(class_map={1: "palm", 0: "bg"})

    assert legend.flag_values == [0, 1]
    assert legend.flag_meanings == "bg palm"
    assert legend.to_attrs()["flag_values"] == [0, 1]
    assert legend.to_attrs()["flag_meanings"] == "bg palm"


def test_class_map_rejects_names_with_whitespace() -> None:
    with pytest.raises(ValidationError, match="whitespace"):
        Legend(class_map={0: "palm oil"})


def test_flag_pair_must_agree_with_class_map() -> None:
    with pytest.raises(ValidationError, match="disagree"):
        Legend(class_map={0: "bg", 1: "palm"}, flag_meanings="bg shrub")


def test_flag_values_and_meanings_are_set_together() -> None:
    with pytest.raises(ValidationError, match="together"):
        Legend(flag_values=[0, 1])


def test_flag_values_must_be_ascending_and_unique() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        Legend(flag_values=[2, 0, 1], flag_meanings="a b c")


def test_a_colour_only_legend_stays_flagless() -> None:
    legend = Legend(color_map={0: "#000000", 1: "#00ff00"})

    assert legend.flag_values is None
    assert legend.flag_meanings is None
