"""Published marks what one stage offers the stages built after it."""

from __future__ import annotations

import pytest
import torch.nn as nn

from geosave_engine.ml.model_chain import Published
from geosave_engine.ml.model_chain.published import (
    accepts,
    published_attrs,
    published_kwargs,
)


class Encoder(nn.Module):
    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]
    unmarked: int

    def __init__(self) -> None:
        super().__init__()
        self.pyramid_channels = [768] * 4
        self.pyramid_strides = [16] * 4
        self.unmarked = 7


class Decoder(nn.Module):
    feature_channels: Published[int]

    def __init__(self) -> None:
        super().__init__()
        self.feature_channels = 256


def test_only_marked_attributes_are_published():
    assert published_attrs(Encoder) == {
        "pyramid_channels": list[int],
        "pyramid_strides": list[int],
    }


def test_a_subclass_inherits_its_base_marks():
    class Bigger(Encoder):
        input_size: Published[int]

    assert set(published_attrs(Bigger)) == {
        "pyramid_channels",
        "pyramid_strides",
        "input_size",
    }


def test_nn_module_own_annotations_are_not_published():
    assert "training" not in published_attrs(Encoder)


def test_a_marked_attribute_holds_a_plain_value_at_runtime():
    assert Encoder().pyramid_channels == [768] * 4


def test_a_parameter_of_the_published_type_is_accepted():
    assert accepts(int, int)


def test_an_optional_parameter_accepts_the_type_it_wraps():
    assert accepts(int, int | None)


def test_a_parameter_of_another_type_is_not_accepted():
    assert not accepts(int, list[int])


def test_a_stage_receives_what_an_earlier_stage_published():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: int) -> None:
            super().__init__()

    assert published_kwargs(Consumer, {"decoder": Decoder()}) == {
        "feature_channels": 256
    }


def test_an_optional_parameter_is_still_wired():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: int | None = None) -> None:
            super().__init__()

    assert published_kwargs(Consumer, {"decoder": Decoder()}) == {
        "feature_channels": 256
    }


def test_a_parameter_nothing_publishes_is_left_alone():
    class Consumer(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()

    assert published_kwargs(Consumer, {"decoder": Decoder()}) == {}


def test_an_unmarked_attribute_is_not_wired_even_when_the_name_matches():
    class Consumer(nn.Module):
        def __init__(self, unmarked: int) -> None:
            super().__init__()

    assert published_kwargs(Consumer, {"encoder": Encoder()}) == {}


def test_the_same_name_at_a_different_type_is_refused():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: list[int]) -> None:
            super().__init__()

    with pytest.raises(TypeError, match="publishes"):
        published_kwargs(Consumer, {"decoder": Decoder()})


def test_two_stages_publishing_one_name_are_refused_by_naming_both():
    class Other(nn.Module):
        feature_channels: Published[int]

        def __init__(self) -> None:
            super().__init__()
            self.feature_channels = 512

    class Consumer(nn.Module):
        def __init__(self, feature_channels: int) -> None:
            super().__init__()

    with pytest.raises(TypeError, match=r"published by \['a', 'b'\]"):
        published_kwargs(Consumer, {"a": Decoder(), "b": Other()})


def test_a_head_reaching_past_the_decoder_to_the_encoder_is_wired():
    class Head(nn.Module):
        def __init__(self, feature_channels: int, pyramid_strides: list[int]) -> None:
            super().__init__()

    resolved = published_kwargs(Head, {"encoder": Encoder(), "decoder": Decoder()})

    assert resolved == {"feature_channels": 256, "pyramid_strides": [16] * 4}


def test_explicit_argument_bypasses_ambiguous_publications():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: int) -> None:
            super().__init__()

    assert published_kwargs(
        Consumer,
        {"a": Decoder(), "b": Decoder()},
        {"feature_channels": 64},
    ) == {"feature_channels": 64}


def test_explicit_argument_bypasses_incompatible_publication():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: list[int]) -> None:
            super().__init__()

    assert published_kwargs(
        Consumer, {"decoder": Decoder()}, {"feature_channels": [64, 128]}
    ) == {"feature_channels": [64, 128]}
