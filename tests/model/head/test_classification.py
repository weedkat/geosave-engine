from __future__ import annotations

import pytest
import torch

from geosave_engine.model.chain import ModelChain
from geosave_engine.model.head.classification import ClassificationHead
from geosave_engine.model.registry import list_models


def _pyramid() -> list[torch.Tensor]:
    return [torch.rand(2, 8, 14, 14), torch.rand(2, 16, 7, 7)]


def test_one_logit_vector_comes_back_per_sample() -> None:
    head = ClassificationHead(classes=["crop", "forest", "urban"], pyramid_channels=[8, 16])

    assert head.num_classes == 3
    assert head.forward_logits(_pyramid()).shape == (2, 3)


def test_the_deepest_level_is_the_one_projected() -> None:
    head = ClassificationHead(classes=["a", "b"], pyramid_channels=[8, 16])

    assert head.projection.in_features == 16


def test_no_classes_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one class"):
        ClassificationHead(classes=[], pyramid_channels=[8])


def test_a_repeated_class_is_refused() -> None:
    with pytest.raises(ValueError, match="'crop'"):
        ClassificationHead(classes=["crop", "crop"], pyramid_channels=[8])


def test_the_head_ends_a_chain_under_its_registered_name() -> None:
    assert "CLASSIFICATION" in list_models("head")["head"]

    chain = ModelChain(head=ClassificationHead(classes=["a", "b"], pyramid_channels=[8, 16]))

    assert chain(pyramid=_pyramid()).shape == (2, 2)
