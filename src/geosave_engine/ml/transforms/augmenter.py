"""Construct native Kornia augmentation from training configuration."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Literal

import kornia.augmentation as K
import torch
from kornia.augmentation.base import _AugmentationBase

DataKey = Literal[
    "input",
    "image",
    "mask",
    "bbox",
    "bbox_xyxy",
    "bbox_xywh",
    "bbox_yolo",
    "keypoints",
    "class",
    "label",
]


def _yolo_to_xyxy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert normalized center/size boxes to pixel corner coordinates."""
    center_x = boxes[..., 0] * width
    center_y = boxes[..., 1] * height
    half_width = boxes[..., 2] * width / 2
    half_height = boxes[..., 3] * height / 2
    return torch.stack(
        [
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ],
        dim=-1,
    )


def _xyxy_to_yolo(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Normalize pixel corner coordinates using the current image size."""
    left, top, right, bottom = boxes.unbind(dim=-1)
    return torch.stack(
        [
            (left + right) / (2 * width),
            (top + bottom) / (2 * height),
            (right - left) / width,
            (bottom - top) / height,
        ],
        dim=-1,
    )


def build_augmentation_pipeline(
    augmentations: list[dict[str, Any]], size: tuple[int, int] | int
) -> list[_AugmentationBase | K.ImageSequential]:
    """Construct Kornia transforms, using the tile size where none is given.

    Nested `AugmentationSequential` configs retain their own constructor options.
    The configuration is left unchanged.

    Raises:
        AttributeError: An augmentation name is absent from Kornia.
    """
    shape = (size, size) if isinstance(size, int) else size
    transforms = []
    for step in augmentations:
        augmentation = getattr(K, step["name"])
        options = dict(step.get("init_args", {}))
        if step["name"] == "AugmentationSequential":
            transforms.append(
                K.AugmentationSequential(
                    *build_augmentation_pipeline(step.get("augmentations", []), shape),
                    **options,
                )
            )
        else:
            if "size" in inspect.signature(augmentation).parameters:
                options.setdefault("size", shape)
            transforms.append(augmentation(**options))
    return transforms


class ImageAugmenter(torch.nn.Module):
    """Apply YAML-configured Kornia augmentation jointly to images and targets.

    Supported data keys:
        - `input`, `image`: Image tensors; the first input must be an image.
        - `mask`: Raster masks, transformed with nearest-neighbor interpolation.
        - `bbox`: Four pixel corner points per box, shaped `(B, N, 4, 2)`.
        - `bbox_xyxy`: Pascal VOC pixel boxes `(xmin, ymin, xmax, ymax)`.
        - `bbox_xywh`: COCO pixel boxes `(xmin, ymin, width, height)`.
        - `bbox_yolo`: Normalized boxes `(cx, cy, width, height)`.
        - `keypoints`: Pixel points `(x, y)`.
        - `class`, `label`: Class labels, passed through unchanged.

    The three four-value box formats use shape `(B, N, 4)`. YOLO class IDs
    belong in a separate `class` or `label` tensor. YOLO boxes are converted
    to pixel `xyxy` before augmentation and normalized using the output image
    size afterward. Box clipping and removal remain the training method's job.

    Args:
        augmentations: Kornia transforms and their constructor arguments.
        size: Default spatial size for transforms such as RandomCrop.

    Raises:
        ValueError: A transform name is unknown.
    """

    def __init__(
        self,
        augmentations: list[dict[str, Any]],
        size: tuple[int, int] | int,
    ) -> None:
        super().__init__()
        self.pipeline = K.AugmentationSequential(
            *build_augmentation_pipeline(augmentations, size),
            data_keys=None,
        )

    def forward(
        self,
        image: torch.Tensor,
        *args: torch.Tensor,
        data_keys: Sequence[DataKey] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply the same augmentation to the supplied image and target tensors.

        Args:
            image: Image whose geometry determines the joint augmentation.
            *args: Other images or targets, in `data_keys` order.
            data_keys: Format of each tensor, required for multiple inputs.
                A single image defaults to `input`.

        Returns:
            One tensor for a single input, otherwise a tuple in input order.
            YOLO boxes remain normalized to the resulting image size.

        Raises:
            ValueError: Multiple inputs lack keys, the key count differs from
                the input count, or the first key is not an image.
        """
        if args and data_keys is None:
            raise ValueError("data_keys is required for multiple inputs.")
        inputs = [image, *args]
        keys = list(data_keys) if data_keys is not None else ["input"]
        if len(keys) != len(inputs):
            raise ValueError("data_keys must match the number of inputs.")
        if keys[0] not in ("input", "image"):
            raise ValueError("The first input must be an image.")

        # Kornia transforms boxes in pixels; only YOLO needs conversion.
        yolo_indices = [index for index, key in enumerate(keys) if key == "bbox_yolo"]
        height, width = image.shape[-2:]
        for index in yolo_indices:
            inputs[index] = _yolo_to_xyxy(inputs[index], height, width)
        native_keys = ["bbox_xyxy" if key == "bbox_yolo" else key for key in keys]
        outputs = self.pipeline(*inputs, data_keys=native_keys)

        if not args:
            return outputs

        # Cropping or resizing can change the image size used for normalization.
        height, width = outputs[0].shape[-2:]
        for index in yolo_indices:
            outputs[index] = _xyxy_to_yolo(outputs[index], height, width)
        return tuple(outputs)
