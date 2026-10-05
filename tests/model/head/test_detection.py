from __future__ import annotations

import pytest
import torch

from geosave_engine.model.chain import ModelChain, Published, chain_step
from geosave_engine.model.head.detection import DetectionHead
from geosave_engine.model.registry import build_model, list_models


def _pyramid() -> list[torch.Tensor]:
    return [torch.rand(2, 8, 14, 14), torch.rand(2, 16, 7, 7)]


def _head() -> DetectionHead:
    return DetectionHead(
        classes=["tree", "building"],
        pyramid_channels=[8, 16],
        pyramid_strides=[8, 16],
        hidden_channels=12,
    )


def test_every_cell_of_every_level_predicts_a_box_and_its_classes() -> None:
    predictions = _head().forward_predictions(_pyramid())

    assert predictions.shape == (2, 14 * 14 + 7 * 7, 4 + 2)


def test_the_strides_are_kept_for_decoding() -> None:
    head = _head()

    assert head.num_classes == 2
    assert head.pyramid_strides == [8, 16]


def test_levels_and_strides_must_pair_up() -> None:
    with pytest.raises(ValueError, match="2 levels.*1 stride"):
        DetectionHead(classes=["tree"], pyramid_channels=[8, 16], pyramid_strides=[8])


def test_no_classes_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one class"):
        DetectionHead(classes=[], pyramid_channels=[8], pyramid_strides=[8])


def test_the_head_ends_a_chain_under_its_registered_name() -> None:
    assert "DETECTION" in list_models("head")["head"]

    chain = ModelChain(head=_head())

    assert chain(pyramid=_pyramid()).shape == (2, 245, 6)


class StubEncoder(torch.nn.Module):
    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]

    def __init__(self) -> None:
        super().__init__()
        self.pyramid_channels = [8, 16]
        self.pyramid_strides = [8, 16]

    @chain_step(outputs=("pyramid",))
    def forward_pyramid(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [torch.rand(image.shape[0], 8, 14, 14), torch.rand(image.shape[0], 16, 7, 7)]


def test_the_encoder_supplies_the_pyramid_layout() -> None:
    model = build_model(
        {
            "encoder": {"class_path": f"{__name__}.StubEncoder"},
            "head": {
                "name": "detection",
                "init_args": {"classes": ["tree"], "hidden_channels": 12},
            },
        }
    )

    assert model.head.pyramid_strides == [8, 16]
    assert model(image=torch.rand(2, 3, 112, 112)).shape == (2, 245, 5)
