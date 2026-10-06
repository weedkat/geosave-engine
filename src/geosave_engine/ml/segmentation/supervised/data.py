from __future__ import annotations

from geosave_engine.geodata.transform.chip import chip_windows
from collections.abc import Sequence

import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from geosave_engine.geodata import read_vector, stack
from stac_geoparquet import to_dict
from geosave_engine.ml.inputs import model_inputs, to_tensor
from geosave_engine.ml.transforms import DataKey, ImageAugmenter
from geosave_engine.model.spec import ModelSpec

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr

    from tiler import Tiler


def _read_target(data: xr.Dataset, ignore_index: int) -> torch.Tensor:
    """Read categorical labels using one nodata-to-ignore conversion."""
    target = to_tensor(data).squeeze(0)
    if target.is_floating_point():
        target = torch.where(torch.isfinite(target), target, ignore_index)
    return target.long()


class Dataset(
    TorchDataset[tuple[dict[str, torch.Tensor], torch.Tensor, str, torch.Tensor]]
):
    """Read a manifest's samples as model inputs and targets, one chip each.

    Every sample is cut as the model spec declares: frames along time, its
    preprocessing on each frame, then chips in space. Every chip of every frame
    is one sample; nothing is drawn at random.

    Args:
        manifest: STAC table whose rows name each sample's rasters as assets.
        spec: Model spec declaring `chips`, `inputs`, and optionally `frames`
            and `preprocessing`.
        target: Layer holding the labels, read raw.
        ignore_index: Label a target pixel takes where its raster has none.

    Raises:
        ValueError: The spec declares no `chips` or no `inputs`, or a row
            lacks a layer the spec or `target` names.

    Examples:
        >>> dataset = Dataset(read_vector("data/train/manifest.parquet"), spec)
        >>> model_inputs, target, tile_id, valid = dataset[0]
        >>> model_inputs["image"].shape, target.shape
        (torch.Size([4, 224, 224]), torch.Size([224, 224]))
    """

    def __init__(
        self,
        manifest: gpd.GeoDataFrame,
        spec: ModelSpec,
        *,
        target: str = "label",
        ignore_index: int = 255,
    ) -> None:
        """Open every sample lazily and number its chips."""
        if spec.chips is None:
            raise ValueError("the model spec declares no chips to cut samples into")
        if not spec.inputs:
            raise ValueError("the model spec declares no inputs for the model")
        self.spec = spec
        self.target = target
        self.ignore_index = ignore_index

        self._layers = (*spec.rasters, target)
        for sample_id, assets in zip(manifest["id"], manifest["assets"], strict=True):
            missing = [name for name in self._layers if name not in assets]
            if missing:
                raise ValueError(
                    f"sample {sample_id!r} has no {missing} layer; its layers are "
                    f"{list(assets)}"
                )
        if not manifest["id"].is_unique or any(
            not isinstance(key, str) or not key for key in manifest["id"]
        ):
            raise ValueError("manifest IDs must be unique non-empty strings")
        self._manifest = manifest

        self._parents: dict[str, xr.DataTree] | None = None
        self._pid: int | None = None
        parents = self._prepare()
        self.reference = self._make_reference(parents)
        del parents
        gc.collect()

    def _make_reference(self, parents: dict[str, xr.DataTree]) -> gpd.GeoDataFrame:
        """Describe prepared windows, annotations and source provenance."""
        assert self.spec.chips is not None
        self.tilers: dict[str, Tiler] = {}
        self.padding: dict[str, list[tuple[int, int]]] = {}
        for key, parent in parents.items():
            y, x = parent.gs.grid_dims
            tiler = self.spec.chips.tiler((parent.sizes[y], parent.sizes[x]))
            self.padding[key] = [(0, 0), (0, 0)]
            if self.spec.chips.overlap:
                padded_shape, self.padding[key] = tiler.calculate_padding()
                tiler.recalculate(data_shape=padded_shape)
            self.tilers[key] = tiler
        reference = chip_windows(parents, self.tilers, padding=self.padding)
        rows = [self._source_rows[key] for key in reference.parent_id]
        reference["source_id"] = [row["id"] for row in rows]
        reference["source_assets"] = [row["assets"] for row in rows]
        # Stored STAC fields describe the raw sample, not these prepared pixels.
        properties = [to_dict(row)["properties"] for row in rows]
        annotations = dict.fromkeys(key for fields in properties for key in fields)
        for column in annotations:
            if column not in reference:
                reference[column] = [fields.get(column) for fields in properties]
        return cast(
            "gpd.GeoDataFrame",
            reference.set_index("id", drop=False, verify_integrity=True),
        )

    def _prepare(self) -> dict[str, xr.DataTree]:
        """Open prepared parents without losing scene or temporal-frame identity."""
        parents = {}
        self._source_rows = {}
        for position in range(len(self._manifest)):
            row = self._manifest.iloc[position]
            source_id = row["id"]
            sample = cast("xr.DataTree", row.gs.to_stack(layers=self._layers))
            frames = (
                (sample,) if self.spec.frames is None else self.spec.frames.cut(sample)
            )
            for index, frame in enumerate(frames):
                parent_id = (
                    source_id
                    if self.spec.frames is None
                    else f"{source_id}/frame-{index}"
                )
                if parent_id in parents:
                    raise ValueError("prepared parent IDs must be unique")
                self._source_rows[parent_id] = row.to_dict()
                rasters = frame.gs.rasters
                results = self.spec.preprocess(rasters)
                parents[parent_id] = stack(
                    {
                        **{name: results[name] for name in self.spec.input_rasters},
                        self.target: rasters[self.target],
                    }
                )
        return parents

    @property
    def parents(self) -> dict[str, xr.DataTree]:
        """Return prepared rasters opened by the current reading process."""
        if self._parents is None or self._pid != os.getpid():
            parents = self._prepare()
            if not self._make_reference(parents).equals(self.reference):
                raise ValueError(
                    "reopened parent frames differ from the reference snapshot"
                )
            self._parents = parents
            self._pid = os.getpid()
        return self._parents

    def __getstate__(self) -> dict[str, object]:
        """Serialize metadata; workers reopen their own parent readers."""
        return {**self.__dict__, "_parents": None, "_pid": None}

    def __len__(self) -> int:
        """Count the chips of every frame of every sample."""
        return len(self.reference)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, str, torch.Tensor]:
        """Read the tile this number names.

        Args:
            index: Tile number in `range(len(self))`.

        Returns:
            ``(model_inputs, target, ids, valid)``. ``model_inputs`` holds the
            inputs the spec declares; target is int64 and validity is boolean,
            both shaped ``(y, x)``. Validity requires finite values across all
            named pixel inputs, including their channels and time axes.

        Raises:
            IndexError: `index` falls outside the cut.
        """
        row = self.reference.iloc[index]
        tile = cast(
            "xr.DataTree",
            row.gs.crop(self.parents[row.parent_id]),
        )
        rasters = tile.gs.rasters
        target = _read_target(rasters[self.target], self.ignore_index)
        inputs = model_inputs(self.spec, rasters, row)
        valid = torch.ones(target.shape, dtype=torch.bool)
        for name in self.spec.pixel_inputs:
            pixels = inputs[name]
            valid &= torch.isfinite(pixels).reshape(-1, *target.shape[-2:]).all(dim=0)
        return inputs, target, row.id, valid


class DataModule(LightningDataModule):
    """Feed supervised segmentation from sample manifests and one model spec.

    Each split is its own manifest. Samples are cut as the model spec declares,
    then every batch is finished on the device: augmentations while training,
    the spec's transforms in every split, and NaN pixels set to zero.

    Args:
        spec: Model spec file declaring `chips`, `inputs`, and `transforms`.
        train: Training manifest.
        val: Validation manifest.
        test: Test manifest. None leaves the test split unavailable.
        target: Layer holding the labels.
        augmentations: Kornia augmentations by ``name`` and ``init_args``, run
            on the pixel inputs and the target together while training.
        ignore_index: Label a target pixel takes where its raster has none or
            an augmentation invented it. Match the module's `ignore_index`.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.

    Examples:
        # LightningCLI YAML:
        data:
          class_path: geosave_engine.ml.segmentation.supervised.DataModule
          init_args:
            spec: configs/model_spec.yaml
            train: data/train/manifest.parquet
            val: data/val/manifest.parquet
            augmentations:
              - {name: RandomHorizontalFlip, init_args: {p: 0.5}}
    """

    def __init__(
        self,
        spec: str | Path,
        train: str | Path,
        val: str | Path,
        *,
        test: str | Path | None = None,
        target: str = "label",
        augmentations: list[dict[str, Any]] | None = None,
        ignore_index: int = 255,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.spec = ModelSpec.load(spec)
        self.manifests = {"train": train, "val": val, "test": test}
        self.target = target
        self.augmentations = augmentations or []
        self.ignore_index = ignore_index
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _dataset(self, split: str) -> Dataset:
        """Read one split's manifest as a dataset."""
        manifest = self.manifests[split]
        if manifest is None:
            raise ValueError(f"no {split} manifest; pass {split}= to the datamodule")
        return Dataset(
            read_vector(manifest),
            self.spec,
            target=self.target,
            ignore_index=self.ignore_index,
        )

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets the stage reads and the batch pipelines.

        Raises:
            ValueError: The stage needs a manifest that was not given, or a
                manifest or the spec cannot feed the model.
        """
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train")
        if stage in (None, "fit", "validate"):
            self.val_dataset = self._dataset("val")
        if stage == "test":
            self.test_dataset = self._dataset("test")

        assert self.spec.chips is not None
        size = self.spec.chips.shape
        self._augmenter = ImageAugmenter(self.augmentations, size)
        self._transforms = {
            name: ImageAugmenter([step.model_dump() for step in steps], size)
            for name, steps in self.spec.transforms.items()
        }

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        # A forked worker inherits open rasters and reader threads it cannot use.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            multiprocessing_context="forkserver" if self.num_workers else None,
            persistent_workers=bool(self.num_workers),
        )

    def train_dataloader(self) -> DataLoader:
        """Return shuffled training batches."""
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation batches in tile order."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return test batches in tile order."""
        return self._loader(self.test_dataset, shuffle=False)

    def on_after_batch_transfer(
        self,
        batch: tuple[
            dict[str, torch.Tensor], torch.Tensor, Sequence[str], torch.Tensor
        ],
        dataloader_idx: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, Sequence[str], torch.Tensor]:
        """Finish a batch on its device.

        Args:
            batch: ``(model_inputs, target, ids, valid)`` as the loaders collate it.
            dataloader_idx: Lightning loader number.

        Returns:
            The batch with its pixel inputs augmented while training,
            transformed, and free of NaN. IDs and context inputs pass through.
            Evaluation validity is intersected with transformed finite pixels.
            Training uses the original validity only as lineage; it is not a
            mask for the augmented geometry and the training module ignores it.
        """
        model_inputs, target, ids, valid = batch
        names = self.spec.pixel_inputs
        images = [model_inputs[name] for name in names]
        if self.trainer is not None and self.trainer.training:
            images, target = self._augment(images, target)

        model_inputs = dict(model_inputs)
        training = self.trainer is not None and self.trainer.training
        for name, image in zip(names, images, strict=True):
            if name in self._transforms:
                # Per-band statistics apply to each instant, so time joins the batch.
                leading = image.shape[:-3]
                image = self._transforms[name](image.flatten(0, -4)).unflatten(
                    0, leading
                )
            if not training:
                valid = valid & torch.isfinite(image).reshape(
                    image.shape[0], -1, *image.shape[-2:]
                ).all(dim=1)
            model_inputs[name] = torch.nan_to_num(image, nan=0.0)
        return model_inputs, target, ids, valid

    def _augment(
        self, images: list[torch.Tensor], target: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the augmentations on every pixel input and the target together."""
        # Time joins the bands, so every instant of a sample moves the same way.
        shapes = [image.shape[1:-2] for image in images]
        # Labels shift up by one, leaving zero for pixels an augmentation invents.
        mask = (target + 1).unsqueeze(1).float()
        keys: list[DataKey] = ["input" for _ in images]
        keys.append("mask")
        *images, mask = self._augmenter(
            *(image.flatten(1, -3) for image in images), mask, data_keys=keys
        )
        target = mask.squeeze(1).round().long() - 1
        target[target < 0] = self.ignore_index
        return [
            image.unflatten(1, shape)
            for image, shape in zip(images, shapes, strict=True)
        ], target
