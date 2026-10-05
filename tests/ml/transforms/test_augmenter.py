import kornia.augmentation as K
import pytest
import torch

from geosave_engine.ml.transforms import ImageAugmenter


def test_yaml_configuration_builds_native_joint_augmentation():
    config = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]
    augmentation = ImageAugmenter(config, size=4, data_keys=["input", "mask"])
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    pixels, labels = augmentation(image, image.clone())
    assert isinstance(augmentation, K.AugmentationSequential)
    torch.testing.assert_close(pixels, image.flip(-1))
    torch.testing.assert_close(labels, image.flip(-1))
    assert config == [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]


def test_nested_yaml_configuration_and_default_crop_size():
    augmentation = ImageAugmenter(
        [
            {
                "name": "AugmentationSequential",
                "augmentations": [{"name": "RandomCrop"}],
            }
        ],
        size=(4, 4),
    )
    assert augmentation(torch.ones(1, 2, 6, 6)).shape == (1, 2, 4, 4)


def test_empty_configuration_is_identity():
    image = torch.ones(1, 2, 4, 4)
    assert ImageAugmenter([], size=4)(image) is image


def test_unknown_augmentation_reports_its_name():
    with pytest.raises(ValueError, match="unknown_transform"):
        ImageAugmenter([{"name": "unknown_transform"}], size=4)
