import pytest
import torch
import torch.nn as nn

from geosave_engine.ml.core.transforms import (
    SemanticSegmentationWrapper,
    build_augmentation_pipeline,
)

IMAGE_SIZE = (4, 4)
BATCH = 2
CHANNELS = 3
NUM_CLASSES = 2


class _Identity(nn.Module):
    """Minimal model that returns input unchanged."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def image() -> torch.Tensor:
    return torch.rand(BATCH, CHANNELS, *IMAGE_SIZE)


@pytest.fixture
def mask() -> torch.Tensor:
    return torch.randint(0, NUM_CLASSES, (BATCH, 1, *IMAGE_SIZE)).float()


class TestBuildAugmentationPipeline:
    def test_empty_list(self):
        result = build_augmentation_pipeline([], IMAGE_SIZE)
        assert result == []

    def test_unknown_aug_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation 'NotReal'"):
            build_augmentation_pipeline([{"name": "NotReal"}], IMAGE_SIZE)

    def test_builds_random_horizontal_flip(self):
        import kornia.augmentation as K
        result = build_augmentation_pipeline(
            [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}],
            IMAGE_SIZE,
        )
        assert len(result) == 1
        assert isinstance(result[0], K.RandomHorizontalFlip)

    def test_injects_size_when_required(self):
        import kornia.augmentation as K
        result = build_augmentation_pipeline(
            [{"name": "RandomCrop"}],
            IMAGE_SIZE,
        )
        assert len(result) == 1
        assert isinstance(result[0], K.RandomCrop)

    def test_nested_augmentation_sequential(self):
        import kornia.augmentation as K
        cfg = [
            {
                "name": "AugmentationSequential",
                "augmentations": [
                    {"name": "RandomHorizontalFlip", "init_args": {"p": 0.5}},
                ],
                "init_args": {},
            }
        ]
        result = build_augmentation_pipeline(cfg, IMAGE_SIZE)
        assert len(result) == 1
        assert isinstance(result[0], K.AugmentationSequential)


class TestSemanticSegmentationWrapper:
    def test_eval_forward_returns_tensor(self, image):
        wrapper = SemanticSegmentationWrapper(_Identity(), augmentations=[], size=IMAGE_SIZE)
        wrapper.eval()
        out = wrapper(image)
        assert isinstance(out, torch.Tensor)
        assert out.shape == image.shape

    def test_train_forward_returns_tuple(self, image, mask):
        wrapper = SemanticSegmentationWrapper(_Identity(), augmentations=[], size=IMAGE_SIZE)
        wrapper.train()
        out = wrapper(image, mask)
        assert isinstance(out, tuple)
        x_out, y_out = out
        assert x_out.shape == image.shape
        assert y_out.shape == mask.shape

    def test_train_no_mask_returns_tensor(self, image):
        wrapper = SemanticSegmentationWrapper(_Identity(), augmentations=[], size=IMAGE_SIZE)
        wrapper.train()
        out = wrapper(image, y=None)
        assert isinstance(out, torch.Tensor)

    def test_none_augmentations_uses_empty_list(self, image, mask):
        wrapper = SemanticSegmentationWrapper(_Identity(), augmentations=None, size=IMAGE_SIZE)
        assert wrapper.augmentations == []

    def test_with_flip_augmentation(self, image, mask):
        augs = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]
        wrapper = SemanticSegmentationWrapper(_Identity(), augmentations=augs, size=IMAGE_SIZE)
        wrapper.train()
        x_out, y_out = wrapper(image, mask)
        assert x_out.shape == image.shape
        assert y_out.shape == mask.shape
