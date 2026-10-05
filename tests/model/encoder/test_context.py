from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from odc.geo.geobox import GeoBox
from torch import nn
from torch.utils.data import default_collate
from transformers import AutoModel

from geosave_engine.geodata import Dataset, GeoVector, raster
from tiler import Tiler
from geosave_engine.geodata.attrs import TimeSpec
from tests.ml.test_inputs import _samples
from geosave_engine.model.encoder import clay, prithvi
from geosave_engine.model.encoder.prithvi import (
    BACKBONE_REGISTRY,
    Prithvi,
    PrithviTL,
)
from geosave_engine.model.registry import build_model
from geosave_engine.model.release.huggingface import GeoSaveModel


@pytest.fixture
def scene() -> Dataset:
    grid = GeoBox.from_bbox((10, 50, 12, 52), "EPSG:4326", shape=(8, 8))
    return raster(
        {"red": np.zeros((2, 8, 8), dtype="float32")},
        grid,
        time=np.array(["2024-12-31T14:00", "2024-01-01T03:00"], dtype="datetime64[m]"),
    )


def _row(data):
    y, x = data.gs.grid_dims
    return GeoVector.from_layouts(
        {"scene": data},
        {
            "scene": Tiler(
                (data.sizes[y], data.sizes[x]), (data.sizes[y], data.sizes[x])
            )
        },
    ).iloc[0]


def test_encoder_model_context_uses_one_row_contract(scene):
    context = prithvi.model_context(_row(scene))
    torch.testing.assert_close(
        context["temporal_coords"], torch.tensor([[2024.0, 365.0], [2024.0, 0.0]])
    )
    torch.testing.assert_close(context["location_coords"], torch.tensor([51.0, 11.0]))
    context = clay.model_context(_row(scene.isel(time=0)))
    torch.testing.assert_close(context["time"], torch.tensor([1.0, 14.0]))
    torch.testing.assert_close(context["latlon"], torch.tensor([51.0, 11.0]))


def test_location_context_accepts_wkt_only_catalog_rows(scene):
    row = _row(scene).drop("proj:code")
    row["proj:wkt2"] = scene.gs.geobox.crs.to_wkt()
    torch.testing.assert_close(prithvi.location_coords(row), torch.tensor([51.0, 11.0]))


def test_prithvi_context_preserves_frame_order_and_calendar_values(
    scene: Dataset,
) -> None:
    times = prithvi.temporal_coords(_row(scene))
    centre = prithvi.location_coords(_row(scene))
    torch.testing.assert_close(times, torch.tensor([[2024.0, 365.0], [2024.0, 0.0]]))
    torch.testing.assert_close(centre, torch.tensor([51.0, 11.0]))
    assert times.dtype == centre.dtype == torch.float32


def test_clay_scalar_time_has_no_frame_axis(scene: Dataset) -> None:
    single = scene.isel(time=0)
    torch.testing.assert_close(clay.time(_row(single)), torch.tensor([1.0, 14.0]))
    torch.testing.assert_close(clay.latlon(_row(single)), torch.tensor([51.0, 11.0]))
    torch.testing.assert_close(clay.time(_row(single["red"])), clay.time(_row(single)))
    torch.testing.assert_close(
        clay.latlon(_row(single["red"])), clay.latlon(_row(single))
    )


def test_context_encodes_bucket_label_not_midpoint(scene: Dataset) -> None:
    monthly = scene.isel(time=slice(1, 2)).gs.rebase(
        TimeSpec.from_resample("MS"), target="time"
    )
    torch.testing.assert_close(
        prithvi.temporal_coords(_row(monthly)), torch.tensor([[2024.0, 0.0]])
    )
    torch.testing.assert_close(clay.time(_row(monthly)), torch.tensor([1.0, 3.0]))


@pytest.mark.parametrize("read_time", [clay.time, prithvi.temporal_coords])
@pytest.mark.parametrize(
    "labels", [[], [np.datetime64("NaT", "ns")], [42], ["invalid"]]
)
def test_invalid_time_labels_raise(scene: Dataset, read_time, labels: list) -> None:
    bad = scene.isel(time=slice(0, len(labels))).assign_coords(time=labels)
    with pytest.raises(ValueError, match="time labels"):
        read_time(_row(bad))


@pytest.mark.parametrize("read_time", [clay.time, prithvi.temporal_coords])
def test_missing_time_raises(scene: Dataset, read_time) -> None:
    with pytest.raises(ValueError, match="time"):
        read_time(_row(scene.isel(time=0, drop=True)))


@pytest.mark.parametrize("read_centre", [clay.latlon, prithvi.location_coords])
def test_missing_grid_raises(read_centre) -> None:
    with pytest.raises(ValueError, match="grid"):
        read_centre(_row(raster({"red": np.zeros((2, 2), dtype="float32")})))


def test_clay_refuses_multiple_frames(scene: Dataset) -> None:
    with pytest.raises(ValueError, match="single frame"):
        clay.time(_row(scene))


def test_prithvi_context_batches_per_tile(scene: Dataset) -> None:
    tiles = _samples({"scene": scene}, (4, 4))
    batch = default_collate(
        [
            {
                "temporal_coords": prithvi.temporal_coords(tiles.reference.iloc[index]),
                "location_coords": prithvi.location_coords(tiles.reference.iloc[index]),
            }
            for index in range(2)
        ]
    )
    assert batch["temporal_coords"].shape == (2, 2, 2)
    torch.testing.assert_close(
        batch["location_coords"], torch.tensor([[51.5, 10.5], [51.5, 11.5]])
    )


def test_clay_context_batches_per_tile(scene: Dataset) -> None:
    tiles = _samples({"scene": scene.isel(time=0)}, (4, 4))
    batch = default_collate(
        [clay.time(tiles.reference.iloc[index]) for index in range(2)]
    )
    torch.testing.assert_close(batch, torch.tensor([[1.0, 14.0], [1.0, 14.0]]))


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


def test_transformers_reload_runs_on_prithvi_context(
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
    tiles = _samples({"scene": scene}, (4, 4))
    inputs = default_collate(
        [
            {
                "image": tiles.read(index).gs.to_tensor(),
                "temporal_coords": prithvi.temporal_coords(tiles.reference.iloc[index]),
                "location_coords": prithvi.location_coords(tiles.reference.iloc[index]),
            }
            for index in range(2)
        ]
    )
    torch.testing.assert_close(
        restored(**inputs),
        model(**inputs),
    )
