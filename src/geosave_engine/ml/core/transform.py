from collections.abc import Callable

import albumentations as A  # noqa: N812 — universally accepted albumentations alias
import numpy as np
import torch


Sample = dict[str, object]


class TransformsCompose:
    """
    Compose a list of transformations specified in the config.

    Callable in two modes:
      - ``tc(image=..., mask=...)`` — classic albumentations kwargs.
      - ``tc(sample)`` — a torchgeo sample dict (image CxHxW tensor, optional
        mask HxW tensor). Bridges tensor<->numpy and writes results back in
        place so it can be used as a step in :class:`TransformPipeline`.
    """
    def __init__(self, cfg=None, input_size=None):

        if not isinstance(cfg, list):
            raise ValueError("Expected a list of transform specifications")

        self.input_size = input_size

        transforms = [self.build_transforms(spec) for spec in cfg]

        # CRS warping in TorchGeo can produce image/mask sizes that differ by a
        # few pixels for the same tile pair. The downstream resize/crop step
        # normalises this, so albumentations' strict shape check is disabled.
        self.transform = A.Compose(transforms, is_check_shapes=False)

    def __call__(self, sample: Sample | None = None, /, **kwargs):
        if sample is None:
            return self.transform(**kwargs)

        image = sample["image"]
        image_np = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
        if image_np.ndim == 4:
            image_np = image_np[0]
        call_kwargs: dict[str, np.ndarray] = {"image": np.moveaxis(image_np, 0, -1)}

        has_mask = "mask" in sample
        if has_mask:
            mask = sample["mask"]
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
            call_kwargs["mask"] = mask_np

        result = self.transform(**call_kwargs)

        out_image = result["image"]
        if not isinstance(out_image, torch.Tensor):
            out_image = torch.from_numpy(np.moveaxis(out_image, -1, 0))
        sample["image"] = out_image.float()

        if has_mask:
            out_mask = result["mask"]
            if not isinstance(out_mask, torch.Tensor):
                out_mask = torch.as_tensor(out_mask)
            sample["mask"] = out_mask.long()
        return sample

    def build_transforms(self, spec):
        name = spec['name']
        args = spec.get('kwargs', {}).copy()
        cls = getattr(A, name)

        if name in ("OneOf", "SomeOf", "Compose"):
            nested_spec = args.pop('transforms', [])
            transforms = [self.build_transforms(t) for t in nested_spec]
            return cls(transforms, **args)

        if name in ("RandomResizedCrop", "RandomCrop", "CenterCrop", "Resize"):
            if self.input_size is not None:
                args['size'] = [self.input_size, self.input_size]
            elif 'size' not in args:
                raise ValueError(f"{name} requires 'size' in kwargs or input_size passed to TransformsCompose")
            # Albumentations splits the API: RandomResizedCrop takes `size`,
            # while RandomCrop/CenterCrop/Resize take separate `height`/`width`.
            if name in ("RandomCrop", "CenterCrop", "Resize"):
                size = args.pop('size')
                height, width = (size, size) if isinstance(size, int) else size
                args.setdefault('height', height)
                args.setdefault('width', width)

        return cls(**args)

    def __add__(self, other):
        if not isinstance(other, TransformsCompose):
            raise ValueError("Can only add another TransformsCompose instance")

        new = TransformsCompose()
        new.transform = A.Compose(self.transform.transforms + other.transform.transforms)
        return new


class TransformPipeline:
    """Sequentially apply sample-level transforms.

    Each step takes a torchgeo sample dict and returns one; compose with
    :func:`remap`, :class:`TransformsCompose`, or any other ``Sample → Sample``
    callable.
    """

    def __init__(self, *steps: Callable[[Sample], Sample]) -> None:
        self._steps = steps

    def __call__(self, sample: Sample) -> Sample:
        for step in self._steps:
            sample = step(sample)
        return sample


def remap(mapping: dict[int, int]) -> Callable[[Sample], Sample]:
    """Return a pipeline step that remaps ``sample['mask']`` values via ``mapping``.

    Values outside ``mapping`` pass through unchanged, so callers should
    supply a complete mapping including any ignore-class entries.
    """
    max_src = max(mapping.keys())
    lut = torch.arange(max_src + 1, dtype=torch.long)
    for src, dst in mapping.items():
        lut[int(src)] = int(dst)

    def _remap(sample: Sample) -> Sample:
        mask = sample["mask"].long()
        clipped = mask.clamp(0, lut.numel() - 1)
        sample["mask"] = lut[clipped]
        return sample

    return _remap


def rename_key(src: str, dst: str) -> Callable[[Sample], Sample]:
    """Return a pipeline step that moves ``sample[src]`` to ``sample[dst]``."""

    def _rename(sample: Sample) -> Sample:
        sample[dst] = sample.pop(src)
        return sample

    return _rename