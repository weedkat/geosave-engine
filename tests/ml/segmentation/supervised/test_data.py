from __future__ import annotations

from geosave_engine.ml.inputs import to_tensor

import gc
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import pystac
import torch
import xarray as xr
from lightning import LightningDataModule
from odc.geo.geobox import GeoBox
from torch.utils.data import DataLoader

from geosave_engine.geodata import GeoVector, raster, read_vector
from geosave_engine.geodata import io
from geosave_engine.ml.segmentation import supervised
from geosave_engine.model.spec import ModelSpec, Ref

SIZE = 20


def test_batch_augmentation_uses_default_crop_size(tmp_path):
    data = _training(_data(tmp_path, augmentations=[{"name": "RandomCrop"}]), True)
    data.setup("fit")
    image, target, _, _ = data.on_after_batch_transfer(_batch(), 0)
    assert image["image"].shape[-2:] == (8, 8)
    assert target.shape[-2:] == (8, 8)


def double(data: xr.Dataset) -> xr.Dataset:
    """Stand in for a preprocessing step that keeps the grid."""
    return data * 2


def location_context(row):
    from geosave_engine.model.encoder.prithvi import location_coords

    return {"location_coords": location_coords(row)}


def _pixels(index: int, times: int = 0) -> np.ndarray:
    plane = np.arange(SIZE * SIZE, dtype="float32").reshape(SIZE, SIZE) + 1000 * index
    return np.stack([plane + day for day in range(times)]) if times else plane


def _manifest(root: Path, *, times: int = 0, label: bool = True) -> gpd.GeoDataFrame:
    """Write two samples and return the manifest listing them."""
    records = []
    for index in range(2):
        left = 500_000 + index * 1000
        grid = GeoBox.from_bbox(
            (left, 9_000_000, left + SIZE * 10, 9_000_000 + SIZE * 10),
            "EPSG:32748",
            resolution=10,
        )
        days = np.array(
            [f"2025-06-{day + 1:02d}" for day in range(times)], "datetime64[ns]"
        )
        optical = raster(
            {"red": _pixels(index, times), "nir": _pixels(index, times) + 0.5},
            grid,
            **({"time": days} if times else {}),
        )
        classes = raster({"class": np.full((SIZE, SIZE), index, "uint8")}, grid)
        folder = root / f"s{index}"
        folder.mkdir(parents=True)
        suffix = ".zarr" if times else ".tif"
        write = io.zarr.write if times else io.geotiff.write_cog
        write(optical, folder / f"optical{suffix}")
        assets = {"optical": folder / f"optical{suffix}"}
        if label or index == 0:
            io.geotiff.write_cog(classes, folder / "label.tif")
            assets["label"] = folder / "label.tif"
        records.append(
            GeoVector.from_items(
                [
                    pystac.Item(
                        id=f"s{index}",
                        geometry=optical.gs.geobox.extent.to_crs("EPSG:4326").json,
                        bbox=list(
                            optical.gs.geobox.extent.to_crs("EPSG:4326").boundingbox
                        ),
                        datetime=datetime(2025, 6, 1, tzinfo=UTC),
                        properties={},
                        assets={
                            name: pystac.Asset(str(href))
                            for name, href in assets.items()
                        },
                    )
                ]
            )
        )
    path = GeoVector.concat(records).gs.to_geoparquet(root / "manifest.parquet")
    # Registering read each sample lazily; collect, so no test starts with them open.
    gc.collect()
    return read_vector(path)


def _spec(**changes: object) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"optical": {"variables": ["red", "nir"]}},
            "preprocessing": {
                "image": {
                    "call": f"{__name__}.double",
                    "kwargs": {"data": Ref("optical")},
                }
            },
            "chips": {"size": 8, "overlap": 2, "window": "hann"},
            "inputs": {
                "image": Ref("image"),
            },
            "context": {
                "call": f"{__name__}.location_context",
                "kwargs": {"row": Ref("row")},
            },
            **changes,
        }
    )


def test_an_input_bound_to_a_raster_reads_the_variables_its_spec_selects(
    tmp_path: Path,
) -> None:
    spec = _spec(
        rasters={"optical": {"variables": ["nir"]}},
        preprocessing={},
        inputs={"image": Ref("optical")},
    )

    model_inputs, _, _, _ = supervised.Dataset(_manifest(tmp_path), spec)[0]

    # The samples hold red and nir; nir alone ends in a half.
    assert model_inputs["image"].shape == (1, 8, 8)
    assert torch.all(model_inputs["image"] % 1 == 0.5)


def test_a_dataset_numbers_every_tile_of_every_sample(tmp_path: Path) -> None:
    dataset = supervised.Dataset(_manifest(tmp_path), _spec())

    # A 20-pixel side takes four 8-pixel tiles stepping by 6, so 16 per sample.
    assert len(dataset) == 32
    assert sum(map(len, dataset.tilers.values())) == 32


def test_a_sample_holds_the_declared_inputs_a_target_and_its_number(
    tmp_path: Path,
) -> None:
    dataset = supervised.Dataset(_manifest(tmp_path), _spec())

    model_inputs, target, index, valid = dataset[21]

    assert index == "s1/tile-5"
    assert valid.shape == target.shape
    assert sorted(model_inputs) == ["image", "location_coords"]
    assert model_inputs["image"].shape == (2, 8, 8)
    assert model_inputs["image"].dtype == torch.float32
    assert model_inputs["location_coords"].shape == (2,)
    assert target.shape == (8, 8)
    assert target.dtype == torch.int64
    assert (target == 1).all()  # tile 21 belongs to the second sample
    # Tile 5 of a sample starts 3 pixels in, past the frame, on both axes.
    expected = 2 * _pixels(1)[3:11, 3:11]
    np.testing.assert_array_equal(model_inputs["image"][0].numpy(), expected)


def test_each_tile_reads_its_own_location(tmp_path: Path) -> None:
    dataset = supervised.Dataset(_manifest(tmp_path), _spec())

    first = dataset[0][0]["location_coords"]
    last = dataset[15][0]["location_coords"]

    assert first[0] > last[0]  # rows run south
    assert first[1] < last[1]  # columns run east


def test_a_target_pixel_the_raster_lacks_is_ignored(tmp_path: Path) -> None:
    spec = _spec(chips={"size": 8, "overlap": 2, "mode": "constant"})
    dataset = supervised.Dataset(_manifest(tmp_path), spec, ignore_index=7)

    model_inputs, target, _, _ = dataset[0]

    assert (target[:3] == 7).all() and (target[3:, 3:] == 0).all()
    assert model_inputs["image"][:, :3].isnan().all()


def test_a_row_lacking_a_layer_is_refused_by_name(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, label=False)

    with pytest.raises(ValueError, match=r"'s1'.*\['label'\]"):
        supervised.Dataset(manifest, _spec())


def test_a_spec_without_chips_or_inputs_is_refused(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="chips"):
        supervised.Dataset(manifest, _spec(chips=None))
    with pytest.raises(ValueError, match="inputs"):
        supervised.Dataset(manifest, _spec(inputs={}))


def test_merged_tiles_rebuild_every_sample(tmp_path: Path) -> None:
    from tiler import Merger

    dataset = supervised.Dataset(_manifest(tmp_path), _spec())
    mergers = {
        key: Merger(layout, logits=2, window="hann", save_visits=False)
        for key, layout in dataset.tilers.items()
    }
    for number in range(len(dataset)):
        model_inputs, _, sample_id, _ = dataset[number]
        row = dataset.reference.loc[sample_id]
        mergers[row.parent_id].add(int(row.tile_id), model_inputs["image"].numpy())
    for index in range(2):
        key = f"s{index}"
        rebuilt = mergers[key].merge(extra_padding=dataset.padding[key])
        np.testing.assert_allclose(rebuilt[0], 2 * _pixels(index), rtol=1e-5)
        labels = to_tensor(dataset.parents[key][dataset.target].dataset)
        assert labels.shape == (1, SIZE, SIZE)
        assert (labels == index).all()


def test_frames_put_a_time_axis_ahead_of_the_bands(tmp_path: Path) -> None:
    spec = _spec(frames={"length": 2, "tolerance": "1D"})
    dataset = supervised.Dataset(_manifest(tmp_path, times=4), spec)

    model_inputs, target, _, _ = dataset[16]

    # Four instants make two frames per sample; tile 16 opens the second frame.
    assert len(dataset) == 2 * 2 * 16
    assert model_inputs["image"].shape == (2, 2, 8, 8)
    assert target.shape == (8, 8)
    np.testing.assert_array_equal(
        model_inputs["image"][:, 0, 3, 3].numpy(), 2 * _pixels(0, 4)[2:, 0, 0]
    )


def _open_rasters() -> list[str]:
    links = [Path("/proc/self/fd") / name for name in os.listdir("/proc/self/fd")]
    return [str(link.resolve()) for link in links if link.resolve().suffix == ".tif"]


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_a_built_dataset_leaves_no_raster_open(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    before = _open_rasters()

    dataset = supervised.Dataset(manifest, _spec())

    assert len(dataset) == 32
    assert _open_rasters() == before


@pytest.mark.slow
def test_worker_processes_read_their_own_rasters(tmp_path: Path) -> None:
    # A worker started fresh imports only installed code, so no test-local call.
    library_call = {
        "image": {
            "call": "geosave_engine.geodata.transform.nodata.to_nan",
            "kwargs": {"data": Ref("optical")},
        }
    }
    dataset = supervised.Dataset(
        _manifest(tmp_path), _spec(preprocessing=library_call, context=None)
    )
    dataset[0]  # this process has read pixels, which a forked worker would hang on

    loader = DataLoader(
        dataset, batch_size=8, num_workers=2, multiprocessing_context="forkserver"
    )
    batches = list(loader)

    numbers = [key for _, _, keys, _ in batches for key in keys]
    images = torch.cat([model_inputs["image"] for model_inputs, _, _, _ in batches])
    assert set(numbers) == set(dataset.reference.id)
    lookup = {key: i for i, key in enumerate(dataset.reference.id)}
    for key, image in zip(numbers, images, strict=True):
        assert torch.equal(image, dataset[lookup[key]][0]["image"])


def _data(tmp_path: Path, spec: ModelSpec | None = None, **options: object):
    _manifest(tmp_path / "train")
    _manifest(tmp_path / "val")
    path = (spec or _spec()).save(tmp_path / "model_spec.yaml")
    return supervised.DataModule(
        spec=path,
        train=tmp_path / "train/manifest.parquet",
        val=tmp_path / "val/manifest.parquet",
        **{"batch_size": 4, "num_workers": 0, **options},
    )


def _batch(
    times: int = 0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    shape = (2, times, 2, 8, 8) if times else (2, 2, 8, 8)
    image = torch.arange(float(np.prod(shape))).reshape(shape)
    target = torch.arange(2 * 8 * 8).reshape(2, 8, 8) % 3
    return (
        {"image": image, "location_coords": torch.ones(2, 2)},
        target,
        ["s0/tile-4", "s0/tile-9"],
        torch.ones((2, 8, 8), dtype=torch.bool),
    )


def _training(data: supervised.DataModule, training: bool) -> supervised.DataModule:
    data.trainer = SimpleNamespace(training=training)  # type: ignore[assignment]
    return data


def test_setup_reads_each_split_from_its_own_manifest(tmp_path: Path) -> None:
    data = _data(tmp_path)

    data.setup("fit")
    model_inputs, target, index, _ = next(iter(data.train_dataloader()))

    assert isinstance(data, LightningDataModule)
    assert len(data.train_dataset) == len(data.val_dataset) == 32
    assert model_inputs["image"].shape == (4, 2, 8, 8)
    assert model_inputs["location_coords"].shape == (4, 2)
    assert target.shape == (4, 8, 8)
    assert len(index) == 4
    assert list(next(iter(data.val_dataloader()))[2]) == [
        f"s0/tile-{i}" for i in range(4)
    ]


def test_the_test_split_needs_its_manifest(tmp_path: Path) -> None:
    data = _data(tmp_path)
    with pytest.raises(ValueError, match="test="):
        data.setup("test")

    data = _data(tmp_path / "again", test=tmp_path / "val/manifest.parquet")
    data.setup("test")
    assert len(data.test_dataloader().dataset) == 32


def test_workers_start_without_this_process_state(tmp_path: Path) -> None:
    data = _data(tmp_path, num_workers=2)
    data.setup("fit")

    loader = data.train_dataloader()

    assert loader.multiprocessing_context.get_start_method() == "forkserver"
    assert loader.persistent_workers
    assert _data(tmp_path / "again").val_dataloader is not None


FLIP = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]
NORMALIZE = {
    "image": [
        {"name": "Normalize", "init_args": {"mean": [1.0, 2.0], "std": [2.0, 4.0]}}
    ]
}


def test_augmentations_move_image_and_target_together_while_training(
    tmp_path: Path,
) -> None:
    data = _training(_data(tmp_path, augmentations=FLIP), True)
    data.setup("fit")
    model_inputs, target, index, valid = _batch()

    augmented, moved, numbers, _ = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(augmented["image"], model_inputs["image"].flip(-1))
    assert torch.equal(moved, target.flip(-1))
    assert moved.dtype == torch.int64
    assert torch.equal(augmented["location_coords"], model_inputs["location_coords"])
    assert numbers == index


def test_augmentations_do_not_run_outside_training(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, augmentations=FLIP), False)
    data.setup("fit")
    model_inputs, target, _, _ = _batch()

    kept, same, _, _ = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(kept["image"], model_inputs["image"])
    assert torch.equal(same, target)


@pytest.mark.parametrize("training", [True, False])
def test_transforms_run_in_every_split_after_augmentation(
    tmp_path: Path, training: bool
) -> None:
    spec = _spec(transforms=NORMALIZE)
    data = _training(_data(tmp_path, spec, augmentations=FLIP), training)
    data.setup("fit")
    image = _batch()[0]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1)

    result = data.on_after_batch_transfer(_batch(), 0)[0]["image"]

    expected = (image.flip(-1) if training else image) - mean
    torch.testing.assert_close(result, expected / std)


def test_pixels_a_raster_lacks_become_zero_after_transforms(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, _spec(transforms=NORMALIZE)), False)
    data.setup("fit")
    model_inputs, target, index, valid = _batch()
    model_inputs["image"][0, :, :2] = torch.nan

    result = data.on_after_batch_transfer((model_inputs, target, index, valid), 0)[0][
        "image"
    ]

    assert not result.isnan().any()
    assert (result[0, :, :2] == 0).all()


def test_a_time_axis_survives_augmentation_and_transforms(tmp_path: Path) -> None:
    spec = _spec(transforms=NORMALIZE)
    data = _training(_data(tmp_path, spec, augmentations=FLIP), True)
    data.setup("fit")
    image = _batch(times=3)[0]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 1, 2, 1, 1)

    result = data.on_after_batch_transfer(_batch(times=3), 0)[0]["image"]

    assert result.shape == (2, 3, 2, 8, 8)
    torch.testing.assert_close(result, (image.flip(-1) - mean) / std)


def test_a_pixel_an_augmentation_invents_is_ignored(tmp_path: Path) -> None:
    shift = [
        {
            "name": "RandomAffine",
            "init_args": {"degrees": 0.0, "translate": [0.5, 0.5], "p": 1.0},
        }
    ]
    data = _training(_data(tmp_path, augmentations=shift, ignore_index=9), True)
    data.setup("fit")
    target = torch.zeros(2, 8, 8, dtype=torch.int64)

    moved = data.on_after_batch_transfer(
        (_batch()[0], target, _batch()[2], _batch()[3]), 0
    )[1]

    assert set(moved.unique().tolist()) == {0, 9}


def test_frames_and_reopened_workers_reproduce_reference_identity(tmp_path):
    import pickle

    spec = _spec(frames={"length": 2, "stride": 1, "tolerance": "1D"})
    manifest = _manifest(tmp_path, times=4)
    dataset = supervised.Dataset(manifest, spec)
    assert set(dataset.parents) == {
        f"s{i}/frame-{j}" for i in range(2) for j in range(3)
    }
    keys = dataset.reference.id.tolist()
    reopened = pickle.loads(pickle.dumps(dataset))
    assert reopened._parents is None
    assert reopened[17][2] == keys[17]
    assert reopened.reference.id.tolist() == keys
    reordered = supervised.Dataset(manifest.iloc[::-1], spec)
    assert set(reordered.reference.id) == set(keys)
    # Same spatial grids, different temporal windows remain distinct parents.
    assert (
        dataset.parents["s0/frame-0"].gs.geobox
        == dataset.parents["s0/frame-1"].gs.geobox
    )


def test_validity_survives_nan_conversion_without_changing_targets(tmp_path):
    data = _training(_data(tmp_path), False)
    data.setup("fit")
    inputs, target, ids, valid = _batch(times=2)
    inputs["image"][0, 1, 0, 3, 4] = torch.nan
    converted, labels, keys, mask = data.on_after_batch_transfer(
        (inputs, target, ids, valid), 0
    )
    assert converted["image"][0, 1, 0, 3, 4] == 0
    assert not mask[0, 3, 4]
    assert mask[1].all()
    assert keys == ids
    torch.testing.assert_close(labels, target)


def test_duplicate_manifest_identity_is_rejected_before_reading(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["id"] = "duplicate"
    with pytest.raises(ValueError, match="unique"):
        supervised.Dataset(manifest, _spec())


@pytest.mark.parametrize(
    "frames", [None, {"length": 2, "stride": 1, "tolerance": "1D"}]
)
def test_prepared_reference_retains_provenance_without_advertising_raw_assets(
    tmp_path, frames
):
    manifest = _manifest(tmp_path, times=4 if frames else 0)
    manifest["class_id"] = [7, 9]
    dataset = supervised.Dataset(manifest, _spec(frames=frames))
    parent_id = "s1/frame-1" if frames else "s1"
    row = dataset.reference.loc[f"{parent_id}/tile-5"]
    assert row.class_id == 9
    assert row.source_id == "s1"
    assert row.source_assets == manifest.iloc[1].assets
    assert "assets" not in row
    assert set(row.raster_metadata) == {"image", "label"}
    with pytest.raises(KeyError, match="assets"):
        row.gs.to_stack()
    tile = row.gs.crop(dataset.parents[row.parent_id])
    assert to_tensor(tile.gs.rasters["label"]).shape == (1, 8, 8)
    expected = (
        2 * _pixels(1, 4)[1:3, 3:11, 3:11] if frames else 2 * _pixels(1)[3:11, 3:11]
    )
    np.testing.assert_array_equal(tile.gs.rasters["image"].red, expected)
    if frames:
        assert row.raster_metadata["image"]["times"] == [
            "2025-06-02T00:00:00.000000000",
            "2025-06-03T00:00:00.000000000",
        ]


def test_evaluation_validity_intersects_named_inputs_and_time_axes(tmp_path):
    spec = _spec(inputs={"image": Ref("optical"), "dem": Ref("optical")})
    data = _training(_data(tmp_path, spec), False)
    data.setup("fit")
    inputs, target, ids, valid = _batch(times=2)
    inputs["dem"] = torch.ones((2, 1, 8, 8))
    inputs["image"][0, 1, 1, 3, 4] = torch.nan
    inputs["dem"][1, 0, 4, 5] = torch.inf
    _, kept, _, mask = data.on_after_batch_transfer((inputs, target, ids, valid), 0)
    assert not mask[0, 3, 4] and not mask[1, 4, 5]
    assert mask[0, 4, 5] and mask[1, 3, 4]
    torch.testing.assert_close(kept, target)
