"""Unit tests for TilePredictionWriter/DensePredictionWriter.

No network — batches/predictions are synthetic GeoTiles/tensors; Lightning's
Trainer is stood in for with a bare stub (only trainer.ckpt_path is read).
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest
import torch
import zarr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.tile import GeoAnchor, GeoStack, GeoTag, GeoTile
from geosave_engine.ml.callbacks.prediction_writer import DensePredictionWriter, TilePredictionWriter

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


class _FakeTrainer:
    ckpt_path = None


class _FakePLModule:
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError


def _tile(bbox=BBOX, *, names=("B02", "B03"), value=1) -> GeoTile:
    gb = GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")
    arr = np.full((len(names), gb.height, gb.width), value, dtype="uint16")
    d = datetime(2024, 1, 15)
    return GeoAnchor(geobox=gb, geotag=GeoTag(datetime=(d, d))).to_geotile(arr, list(names))


def _batch(n: int = 1, **tiles: GeoTile) -> dict:
    """A batch dict carrying only batch["tiles"] — the only key either writer reads."""
    return {"tiles": {key: [tile] * n for key, tile in tiles.items()}}


class TestTilePredictionWriter:
    def test_requires_write_zarr_or_write_cog(self, tmp_path):
        with pytest.raises(ValueError, match="write_zarr or write_cog"):
            TilePredictionWriter(tmp_path, "model", write_zarr=False, write_cog=False)

    def test_writes_every_batch_tiles_layer(self, tmp_path):
        writer = TilePredictionWriter(tmp_path, "model", job_id="job1")
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        tile = _tile(names=("B02", "B03"))
        batch = _batch(sentinel_2_l1c=tile)
        writer.write_on_batch_end(_FakeTrainer(), None, None, None, batch, 0, 0)

        store = tmp_path / "model" / "job1" / "tiles" / f"{tile.stem}.zarr"
        assert sorted(zarr.open_group(store, mode="r").group_keys()) == ["sentinel_2_l1c"]

    def test_layers_filter_persists_only_named_keys(self, tmp_path):
        writer = TilePredictionWriter(tmp_path, "model", job_id="job1", layers=["a"])
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        batch = _batch(a=_tile(names=("B02",)), b=_tile(names=("B02",)))
        writer.write_on_batch_end(_FakeTrainer(), None, None, None, batch, 0, 0)

        stem = batch["tiles"]["a"][0].stem
        store = tmp_path / "model" / "job1" / "tiles" / f"{stem}.zarr"
        assert sorted(zarr.open_group(store, mode="r").group_keys()) == ["a"]

    def test_adds_rgb_subset_companion_when_derivable(self, tmp_path):
        tile = _tile(names=("red", "green", "blue")).rebase(plot_meta={"rgb_bands": ("red", "green", "blue")})
        writer = TilePredictionWriter(tmp_path, "model", job_id="job1")
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        batch = _batch(image=tile)
        writer.write_on_batch_end(_FakeTrainer(), None, None, None, batch, 0, 0)

        store = tmp_path / "model" / "job1" / "tiles" / f"{tile.stem}.zarr"
        assert sorted(zarr.open_group(store, mode="r").group_keys()) == ["image", "image_rgb"]

    def test_skips_already_written_layers(self, tmp_path, caplog):
        writer = TilePredictionWriter(tmp_path, "model", job_id="job1")
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        batch = _batch(a=_tile(names=("B02",)))
        writer.write_on_batch_end(_FakeTrainer(), None, None, None, batch, 0, 0)
        with caplog.at_level("INFO"):
            writer.write_on_batch_end(_FakeTrainer(), None, None, None, batch, 0, 0)
        assert "already written" in caplog.text

    def test_writes_metadata_json(self, tmp_path):
        writer = TilePredictionWriter(tmp_path, "model", job_id="job1")
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        writer.on_predict_end(_FakeTrainer(), None)
        metadata = json.loads((tmp_path / "model" / "job1" / "metadata.json").read_text())
        assert metadata["model_name"] == "model"
        assert metadata["job_id"] == "job1"


class TestDensePredictionWriter:
    def test_requires_write_zarr_or_write_cog(self, tmp_path):
        with pytest.raises(ValueError, match="write_zarr or write_cog"):
            DensePredictionWriter(tmp_path, "model", image_key="image", write_zarr=False, write_cog=False)

    def test_raises_when_no_recognized_key_present(self, tmp_path):
        writer = DensePredictionWriter(tmp_path, "model", image_key="image", job_id="job1")
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        batch = _batch(image=_tile(names=("B02",)))
        with pytest.raises(KeyError, match="needs at least one of"):
            writer.write_on_batch_end(_FakeTrainer(), None, {"debug": torch.zeros(1, 32, 32)}, None, batch, 0, 0)

    def test_writes_recognized_keys_as_tiles(self, tmp_path):
        writer = DensePredictionWriter(tmp_path, "model", image_key="image", job_id="job1")
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        tile = _tile(names=("B02", "B03"))
        batch = _batch(image=tile)
        prediction = {"pred": torch.zeros(1, 32, 32), "proba": torch.ones(1, 32, 32)}
        writer.write_on_batch_end(_FakeTrainer(), None, prediction, None, batch, 0, 0)

        store = tmp_path / "model" / "job1" / "tiles" / f"{tile.stem}.zarr"
        assert sorted(zarr.open_group(store, mode="r").group_keys()) == ["pred", "proba"]

    def test_bakes_class_map_and_color_map_into_plot_meta(self, tmp_path):
        writer = DensePredictionWriter(
            tmp_path, "model", image_key="image", job_id="job1",
            class_map={0: "water", 1: "trees"}, color_map={0: "#0000ff", 1: "#00ff00"},
        )
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        tile = _tile(names=("B02",))
        batch = _batch(image=tile)
        writer.write_on_batch_end(_FakeTrainer(), None, {"pred": torch.zeros(1, 32, 32)}, None, batch, 0, 0)

        store = tmp_path / "model" / "job1" / "tiles" / f"{tile.stem}.zarr"
        loaded = GeoStack.load(store, load_data=True)
        assert loaded.tiles["pred"].plot_meta.class_map == {0: "water", 1: "trees"}
        assert loaded.tiles["pred"].plot_meta.color_map == {0: "#0000ff", 1: "#00ff00"}
        assert loaded.tiles["pred"].metadata["model_name"] == "model"
        assert loaded.tiles["pred"].metadata["job_id"] == "job1"

    def test_source_tile_not_tagged_with_prediction_provenance(self, tmp_path):
        writer = DensePredictionWriter(tmp_path, "model", image_key="image", job_id="job1")
        writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        tile = _tile(names=("B02",))
        batch = _batch(image=tile)
        writer.write_on_batch_end(_FakeTrainer(), None, {"pred": torch.zeros(1, 32, 32)}, None, batch, 0, 0)
        assert "model_name" not in tile.metadata


class TestComposability:
    """TilePredictionWriter and DensePredictionWriter pointed at the same run
    directory must not clobber each other, regardless of which runs first."""

    def _run(self, tmp_path, *, tile_first: bool):
        image = _tile(names=("B02", "B03"))
        batch = _batch(image=image)
        prediction = {"pred": torch.zeros(1, 32, 32)}

        tile_writer = TilePredictionWriter(tmp_path, "model", job_id="job1")
        dense_writer = DensePredictionWriter(tmp_path, "model", image_key="image", job_id="job1")
        tile_writer.setup(_FakeTrainer(), _FakePLModule(), "predict")
        dense_writer.setup(_FakeTrainer(), _FakePLModule(), "predict")

        writers = [tile_writer, dense_writer] if tile_first else [dense_writer, tile_writer]
        for writer in writers:
            args = (None, batch) if isinstance(writer, TilePredictionWriter) else (prediction, batch)
            writer.write_on_batch_end(_FakeTrainer(), None, args[0], None, args[1], 0, 0)

        return tmp_path / "model" / "job1" / "tiles" / f"{image.stem}.zarr"

    def test_tile_then_dense(self, tmp_path):
        store = self._run(tmp_path, tile_first=True)
        assert sorted(zarr.open_group(store, mode="r").group_keys()) == ["image", "pred"]

    def test_dense_then_tile(self, tmp_path):
        store = self._run(tmp_path, tile_first=False)
        assert sorted(zarr.open_group(store, mode="r").group_keys()) == ["image", "pred"]
