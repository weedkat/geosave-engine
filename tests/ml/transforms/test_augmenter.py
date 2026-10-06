import kornia.augmentation as K
import pytest
import torch

from geosave_engine.ml.transforms import ImageAugmenter


def test_yaml_configuration_builds_native_joint_augmentation():
    config = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]
    augmentation = ImageAugmenter(config, size=4)
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    pixels, labels = augmentation(image, image.clone(), data_keys=["input", "mask"])
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
    with pytest.raises(AttributeError, match="unknown_transform"):
        ImageAugmenter([{"name": "unknown_transform"}], size=4)


def test_yolo_boxes_flip_with_the_image_without_mutating_input():
    augmentation = ImageAugmenter(
        [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}],
        size=(4, 8),
    )
    image = torch.arange(32, dtype=torch.float32).reshape(1, 1, 4, 8)
    boxes = torch.tensor([[[0.25, 0.5, 0.25, 0.5]]])
    labels = torch.tensor([[2]])
    original = boxes.clone()

    pixels, predictions, classes = augmentation(
        image, boxes, labels, data_keys=["image", "bbox_yolo", "label"]
    )

    torch.testing.assert_close(pixels, image.flip(-1))
    # Kornia reflects pixel coordinates around width - 1.
    torch.testing.assert_close(predictions, torch.tensor([[[0.625, 0.5, 0.25, 0.5]]]))
    torch.testing.assert_close(boxes, original)
    torch.testing.assert_close(classes, labels)


def test_yolo_boxes_use_output_size_after_resize_and_mix_with_native_boxes():
    augmentation = ImageAugmenter(
        [{"name": "Resize", "init_args": {"size": (8, 16)}}],
        size=(4, 8),
    )
    image = torch.zeros(1, 1, 4, 8)
    yolo = torch.tensor([[[0.25, 0.5, 0.25, 0.5]]])
    xyxy = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
    native = K.AugmentationSequential(
        K.Resize((8, 16)), data_keys=["input", "bbox_xyxy"]
    )
    expected_image, expected_xyxy = native(image, xyxy)
    minimum, maximum = expected_xyxy[..., :2], expected_xyxy[..., 2:]
    output_size = torch.tensor([16, 8])
    expected_yolo = torch.cat(
        [(minimum + maximum) / 2 / output_size, (maximum - minimum) / output_size],
        dim=-1,
    )

    pixels, first, native_boxes, second = augmentation(
        image,
        yolo,
        xyxy,
        yolo,
        data_keys=["input", "bbox_yolo", "bbox_xyxy", "bbox_yolo"],
    )

    torch.testing.assert_close(pixels, expected_image)
    torch.testing.assert_close(first, expected_yolo)
    torch.testing.assert_close(native_boxes, expected_xyxy)
    torch.testing.assert_close(second, expected_yolo)


@pytest.mark.parametrize("box_key", ["bbox", "bbox_xyxy", "bbox_xywh"])
def test_native_box_formats_and_keypoints_match_kornia(box_key):
    image = torch.zeros(1, 1, 4, 8)
    boxes = {
        "bbox": torch.tensor([[[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]]]),
        "bbox_xyxy": torch.tensor([[[1.0, 1.0, 3.0, 3.0]]]),
        "bbox_xywh": torch.tensor([[[1.0, 1.0, 2.0, 2.0]]]),
    }[box_key]
    points = torch.tensor([[[1.0, 1.0]]])
    keys = ["input", box_key, "keypoints", "class"]
    labels = torch.tensor([[2]])
    augmentation = ImageAugmenter(
        [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}],
        size=(4, 8),
    )
    native = K.AugmentationSequential(K.RandomHorizontalFlip(p=1), data_keys=keys)

    actual = augmentation(image, boxes, points, labels, data_keys=keys)
    expected = native(image, boxes, points, labels)

    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)


def test_yolo_requires_an_image_to_determine_pixel_size():
    augmentation = ImageAugmenter([], size=4)
    with pytest.raises(ValueError, match="first.*image"):
        augmentation(torch.ones(1, 1, 4), data_keys=["bbox_yolo"])


def test_yolo_empty_pipeline_preserves_boxes():
    augmentation = ImageAugmenter([], size=4)
    image = torch.ones(1, 1, 4, 4)
    boxes = torch.tensor([[[0.25, 0.5, 0.25, 0.5]]])

    pixels, result = augmentation(image, boxes, data_keys=["input", "bbox_yolo"])

    assert pixels is image
    torch.testing.assert_close(result, boxes)


def test_keys_can_switch_between_yolo_native_boxes_and_image_only():
    augmentation = ImageAugmenter(
        [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}],
        size=(4, 8),
    )
    image = torch.zeros(1, 1, 4, 8)
    boxes = torch.tensor([[[0.25, 0.5, 0.25, 0.5]]])
    xyxy = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
    _, yolo_output = augmentation(image, boxes, data_keys=["input", "bbox_yolo"])

    _, native_output = augmentation(image, xyxy, data_keys=["input", "bbox_xyxy"])
    _, yolo_again = augmentation(image, boxes, data_keys=["input", "bbox_yolo"])
    image_only = augmentation(image)

    torch.testing.assert_close(native_output, torch.tensor([[[4.0, 1.0, 6.0, 3.0]]]))
    torch.testing.assert_close(yolo_again, yolo_output)
    torch.testing.assert_close(image_only, image.flip(-1))


def test_multiple_inputs_require_explicit_keys():
    augmentation = ImageAugmenter([], size=4)
    image = torch.ones(1, 1, 4, 4)

    with pytest.raises(ValueError, match="data_keys.*multiple inputs"):
        augmentation(image, image.clone())


@pytest.mark.parametrize("keys", [[], ["input"], ["input", "mask", "class"]])
def test_keys_must_match_the_number_of_inputs(keys):
    augmentation = ImageAugmenter([], size=4)
    image = torch.ones(1, 1, 4, 4)

    with pytest.raises(ValueError, match="data_keys.*number of inputs"):
        augmentation(image, image.clone(), data_keys=keys)
