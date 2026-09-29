from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from odc.geo.geobox import GeoBox
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from geosave_engine.geodata import DataTree, Dataset, raster, stack
from geosave_engine.geodata.attrs import TimeSpec
from geosave_engine.geodata.datasets import TileDataset
from geosave_engine.geodata.transform.tiling import Tiles
from geosave_engine.release.huggingface import GeoSaveModel
from geosave_engine.ml.registry import build_model
from geosave_engine.ml.models.encoder.clay import Clay
from geosave_engine.ml.models.encoder.prithvi import (
    BACKBONE_REGISTRY,
    Prithvi,
    PrithviTL,
)


@pytest.fixture
def scene() -> Dataset:
    grid = GeoBox.from_bbox((10, 50, 12, 52), "EPSG:4326", shape=(8, 8))
    return raster(
        {"red": np.zeros((2, 8, 8), dtype="float32")},
        grid,
        time=np.array(["2024-12-31T14:00", "2024-01-01T03:00"], dtype="datetime64[m]"),
    )


def test_prithvi_context_preserves_frame_order_and_calendar_values(
    scene: Dataset,
) -> None:
    context = PrithviTL.model_context(scene)
    torch.testing.assert_close(
        context["temporal_coords"], torch.tensor([[2024.0, 365.0], [2024.0, 0.0]])
    )
    torch.testing.assert_close(context["location_coords"], torch.tensor([51.0, 11.0]))
    assert all(value.dtype == torch.float32 for value in context.values())


def test_clay_scalar_time_has_no_frame_axis(scene: Dataset) -> None:
    single = scene.isel(time=0)
    context = Clay.model_context(single)
    torch.testing.assert_close(context["time"], torch.tensor([1.0, 14.0]))
    torch.testing.assert_close(context["latlon"], torch.tensor([51.0, 11.0]))
    band_context = Clay.model_context(single["red"])
    torch.testing.assert_close(band_context, context)


def test_context_encodes_bucket_label_not_midpoint(scene: Dataset) -> None:
    monthly = scene.isel(time=slice(1, 2)).gs.rebase(
        TimeSpec.from_resample("MS"), target="time"
    )
    torch.testing.assert_close(
        PrithviTL.model_context(monthly)["temporal_coords"],
        torch.tensor([[2024.0, 0.0]]),
    )
    torch.testing.assert_close(
        Clay.model_context(monthly)["time"], torch.tensor([1.0, 3.0])
    )


@pytest.mark.parametrize("encoder", [Clay, PrithviTL])
@pytest.mark.parametrize(
    "labels", [[], [np.datetime64("NaT", "ns")], [42], ["invalid"]]
)
def test_invalid_time_labels_raise(
    scene: Dataset, encoder: type[Clay] | type[PrithviTL], labels: list
) -> None:
    bad = scene.isel(time=slice(0, len(labels))).assign_coords(time=labels)
    with pytest.raises(ValueError, match="time labels"):
        encoder.model_context(bad)


@pytest.mark.parametrize("encoder", [Clay, PrithviTL])
def test_missing_time_or_grid_raises(
    scene: Dataset, encoder: type[Clay] | type[PrithviTL]
) -> None:
    with pytest.raises(ValueError, match="'time'"):
        encoder.model_context(scene.isel(time=0, drop=True))
    with pytest.raises(ValueError, match="grid"):
        encoder.model_context(
            raster({"red": np.zeros((2, 2), dtype="float32")}).assign_coords(
                time=np.datetime64("2024-01-01", "ns")
            )
        )


def test_clay_refuses_multiple_frames(scene: Dataset) -> None:
    with pytest.raises(ValueError, match="single frame"):
        Clay.model_context(scene)


def test_tile_context_collates_with_frame_and_batch_axes(scene: Dataset) -> None:
    samples = TileDataset(Tiles([scene], (4, 4)), model_context=PrithviTL.model_context)
    inputs, _ = next(iter(DataLoader(samples, batch_size=2)))
    assert inputs["image"].shape == (2, 2, 1, 4, 4)
    assert inputs["temporal_coords"].shape == (2, 2, 2)
    assert inputs["location_coords"].shape == (2, 2)
    torch.testing.assert_close(
        inputs["location_coords"],
        torch.tensor([[51.5, 10.5], [51.5, 11.5]]),
    )


def test_clay_tile_context_batches_single_frame(scene: Dataset) -> None:
    samples = TileDataset(
        Tiles([scene.isel(time=0)], (4, 4)), model_context=Clay.model_context
    )
    inputs, _ = next(iter(DataLoader(samples, batch_size=2)))
    assert inputs["image"].shape == (2, 1, 4, 4)
    torch.testing.assert_close(
        inputs["time"], torch.tensor([[1.0, 14.0], [1.0, 14.0]])
    )


class PatchEmbed(nn.Module):
    patch_size = (1, 2, 2)


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.out_channels = [3] * 12
        self.patch_embed = PatchEmbed()
        self.received: torch.Tensor | None = None
        self.context: dict[str, torch.Tensor | None] = {}

    def forward(
        self, image: torch.Tensor, **context: torch.Tensor | None
    ) -> list[torch.Tensor]:
        return self.forward_features(image, **context)

    def forward_features(
        self, image: torch.Tensor, **context: torch.Tensor | None
    ) -> list[torch.Tensor]:
        self.received = image
        self.context = context
        return [image.flatten(2).transpose(1, 2) * self.scale]

    def prepare_features_for_image_model(
        self, features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return features


@pytest.mark.parametrize("encoder", [Prithvi, PrithviTL])
@pytest.mark.parametrize("temporal", [False, True])
def test_prithvi_reorders_geodata_temporal_pixels(
    monkeypatch: pytest.MonkeyPatch,
    encoder: type[Prithvi] | type[PrithviTL],
    temporal: bool,
) -> None:
    backbone = Backbone()

    def build(*args: object, **kwargs: object) -> Backbone:
        return backbone

    monkeypatch.setattr(BACKBONE_REGISTRY, "build", build)
    model = encoder(out_indices=[0], num_frames=2 if temporal else 1)
    shape = (2, 2, 3, 4, 4) if temporal else (2, 3, 4, 4)
    image = torch.arange(np.prod(shape).item(), dtype=torch.float32).reshape(shape)
    expected = image.transpose(1, 2) if temporal else image
    if isinstance(model, PrithviTL):
        times = torch.zeros(2, 2 if temporal else 1, 2)
        locations = torch.zeros(2, 2)
        model.forward_pyramid(image, temporal_coords=times, location_coords=locations)
        assert backbone.context["temporal_coords"] is times
        assert backbone.context["location_coords"] is locations
    else:
        model.forward_pyramid(image)
    torch.testing.assert_close(backbone.received, expected)
    model(image)
    torch.testing.assert_close(backbone.received, expected)


def test_stack_context_selects_its_source_explicitly(scene: Dataset) -> None:
    grouped = stack({"optical": scene, "dem": scene.isel(time=0, drop=True)})

    def optical_context(tile: DataTree) -> dict[str, torch.Tensor]:
        return PrithviTL.model_context(tile["optical"].dataset)

    samples = TileDataset(Tiles([grouped], (4, 4)), model_context=optical_context)
    inputs, _ = next(iter(DataLoader(samples, batch_size=2)))
    assert sorted(inputs["image"]) == ["dem", "optical"]
    torch.testing.assert_close(
        inputs["temporal_coords"],
        torch.tensor([[[2024.0, 365.0], [2024.0, 0.0]]] * 2),
    )


def test_transformers_reload_restores_prithvi_context_extractor(
    scene: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build(*args: object, **kwargs: object) -> Backbone:
        return Backbone()

    monkeypatch.setattr(BACKBONE_REGISTRY, "build", build)
    model = build_model(
        {
            "encoder": {
                "name": "prithvi_tl",
                "init_args": {
                    "out_indices": [0],
                    "num_frames": 2,
                    "in_channels": 1,
                    "input_size": 8,
                },
            }
        }
    )
    original_encoder = model.get_submodule("encoder")
    assert isinstance(original_encoder, PrithviTL)
    original_backbone = original_encoder.model
    assert isinstance(original_backbone, Backbone)
    with torch.no_grad():
        original_backbone.scale.fill_(7.0)
    GeoSaveModel.from_chain(model).save_pretrained(tmp_path)

    restored = AutoModel.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(restored, GeoSaveModel)
    restored_encoder = restored.chain.get_submodule("encoder")
    assert isinstance(restored_encoder, PrithviTL)
    assert restored_encoder.model is not original_backbone
    torch.testing.assert_close(
        restored_encoder.model.state_dict()["scale"], torch.tensor(7.0)
    )
    samples = TileDataset(
        Tiles([scene], (4, 4)), model_context=restored_encoder.model_context
    )
    inputs, _ = next(iter(DataLoader(samples, batch_size=2)))
    torch.testing.assert_close(
        restored(**inputs),
        model(**inputs),
    )
