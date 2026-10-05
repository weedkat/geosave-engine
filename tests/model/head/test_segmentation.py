from __future__ import annotations

import json

import pytest
import torch

from geosave_engine.model.chain import ModelChain, Published, chain_step
from geosave_engine.model.head.segmentation import SegmentationHead
from geosave_engine.model.registry import build_model, list_models


def test_the_classes_fix_the_channel_count() -> None:
    head = SegmentationHead(classes=["background", "oil_palm", "water"], feature_channels=4)

    logits = head.forward_logits(torch.rand(2, 4, 8, 8))

    assert head.num_classes == 3
    assert logits.shape == (2, 3, 8, 8)


def test_logits_are_resized_to_the_input_size() -> None:
    head = SegmentationHead(classes=["a", "b"], feature_channels=4, input_size=16)

    assert head.forward_logits(torch.rand(1, 4, 14, 14)).shape == (1, 2, 16, 16)


def test_no_classes_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one class"):
        SegmentationHead(classes=[], feature_channels=4)


def test_a_repeated_class_is_refused() -> None:
    with pytest.raises(ValueError, match="'water'"):
        SegmentationHead(classes=["water", "forest", "water"], feature_channels=4)


def test_the_head_ends_a_chain() -> None:
    chain = ModelChain(head=SegmentationHead(classes=["a", "b"], feature_channels=4))

    assert chain(feature_map=torch.rand(1, 4, 8, 8)).shape == (1, 2, 8, 8)


def test_the_registered_name_builds_a_head_whose_classes_survive_json() -> None:
    assert "SEGMENTATION" in list_models("head")["head"]

    model = build_model(
        {
            "head": {
                "name": "segmentation",
                "init_args": {"classes": ["background", "oil_palm"], "feature_channels": 4},
            }
        }
    )

    saved = json.loads(json.dumps(model.stage_specs))
    assert saved["head"]["init_args"]["classes"] == ["background", "oil_palm"]
    assert model.head.classes == ["background", "oil_palm"]


class StubEncoder(torch.nn.Module):
    input_size: Published[int | tuple[int, int]]

    def __init__(self) -> None:
        super().__init__()
        self.input_size = 16

    @chain_step(outputs=("pyramid",))
    def forward_pyramid(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [torch.rand(image.shape[0], 4, 14, 14)]


class StubDecoder(torch.nn.Module):
    feature_channels: Published[int]

    def __init__(self) -> None:
        super().__init__()
        self.feature_channels = 4

    @chain_step(outputs=("feature_map",))
    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        return pyramid[0]


def test_earlier_stages_supply_the_feature_width_and_input_size() -> None:
    model = build_model(
        {
            "encoder": {"class_path": f"{__name__}.StubEncoder"},
            "decoder": {"class_path": f"{__name__}.StubDecoder"},
            "head": {"name": "segmentation", "init_args": {"classes": ["a", "b", "c"]}},
        }
    )

    assert model(image=torch.rand(2, 3, 16, 16)).shape == (2, 3, 16, 16)
