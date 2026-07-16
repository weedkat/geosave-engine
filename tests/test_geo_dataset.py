"""Unit tests for GeoDataset: anchor-folder discovery, save_dataset, stream_ingest.

No network — tiles are synthetic zarr stores written to a tmp folder.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.tile import GEOSTACK_SUFFIX, GeoAnchor, GeoTile, GeoStack
from geosave_engine.geodata.datasets import GeoDataset, stack_samples
from geosave_engine.geodata.pipeline import GeoPipeline, save_dataset, stream_ingest

UTM = "EPSG:32633"


def _write_anchor(root: Path, idx: int, bbox, layers: dict[str, tuple[str, ...]]) -> None:
    """Write one anchor folder under root, with the given layers (name -> band names)."""
    gb = GeoBox.from_bbox(bbox, crs=UTM, resolution=10, anchor="edge")
    tiles = {}
    for name, band_names in layers.items():
        arr = np.zeros((len(band_names), gb.height, gb.width), dtype="uint16")
        tiles[name] = GeoAnchor(geobox=gb, datetime=datetime(2023, 2, 1)).with_np(arr, list(band_names))
    GeoStack(**tiles).save(root / f"t{idx}{GEOSTACK_SUFFIX}")


def _dataset_root(root: Path, n_anchors: int = 2) -> Path:
    """Write n_anchors anchor folders (s2 + label layers) under root."""
    for i in range(n_anchors):
        x0 = 500000 + i * 320
        bbox = (x0, 5000000, x0 + 320, 5000320)
        _write_anchor(root, i, bbox, {"s2": ("B02", "B03"), "label": ("label",)})
    return root


class TestDiscovery:
    def test_finds_every_anchor_folder(self, tmp_path):
        ds = GeoDataset(_dataset_root(tmp_path, n_anchors=2))
        assert sorted(ds.layers) == ["label", "s2"]
        assert len(ds) == 2

    def test_getitem_renders_tensors(self, tmp_path):
        ds = GeoDataset(_dataset_root(tmp_path))
        sample = ds[0]
        assert isinstance(sample["s2"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)
        assert tuple(sample["s2"].shape) == (2, 32, 32)   # 2 bands, H, W
        assert tuple(sample["label"].shape) == (1, 32, 32)

    def test_context_override(self, tmp_path):
        def my_context(tiles):
            ref = next(iter(tiles.values()))
            return {"bbox": torch.tensor(ref.bbox)}

        ds = GeoDataset(_dataset_root(tmp_path), context_fn=my_context)
        sample = ds[0]
        assert "bbox" in sample["context"]
        assert isinstance(sample["context"]["bbox"], torch.Tensor)

    def test_single_layer(self, tmp_path):
        _write_anchor(tmp_path, 0, (500000, 5000000, 500320, 5000320), {"s2": ("B02",)})
        ds = GeoDataset(tmp_path)
        assert ds.layers == ["s2"]
        assert len(ds) == 1

    def test_required_layers_excludes_incomplete_anchors(self, tmp_path):
        _write_anchor(tmp_path, 0, (500000, 5000000, 500320, 5000320), {"s2": ("B02",), "label": ("label",)})
        _write_anchor(tmp_path, 1, (500320, 5000000, 500640, 5000320), {"s2": ("B02",)})  # no label
        ds = GeoDataset(tmp_path, required_layers=["s2", "label"])
        assert len(ds) == 1

    def test_finds_anchors_nested_arbitrarily_deep(self, tmp_path):
        """Anchor folders can be grouped in nested subdirectories (e.g. mirroring raw
        data provenance), not just flat directly under root — discovery is by
        .geostack marker via rglob, any depth."""
        _write_anchor(tmp_path / "Experts" / "EH" / "1", 0, (500000, 5000000, 500320, 5000320), {"s2": ("B02",)})
        _write_anchor(tmp_path / "Non_expert" / "WorkForce" / "EH" / "1", 0,
                      (500320, 5000000, 500640, 5000320), {"s2": ("B02",)})
        ds = GeoDataset(tmp_path)
        assert len(ds) == 2


class TestRender:
    def test_stack_samples_batches_per_layer_key(self, tmp_path):
        ds = GeoDataset(_dataset_root(tmp_path), sel_bands={"s2": ["B02"]})
        batch = stack_samples([ds[0], ds[1]])
        assert tuple(batch["s2"].shape) == (2, 1, 32, 32)   # B, C(sel B02), H, W
        assert tuple(batch["label"].shape) == (2, 1, 32, 32)
        assert len(batch["s2"]) == 2


class _ToyPipeline(GeoPipeline):
    def ingest(self, anchor: GeoAnchor) -> dict[str, GeoTile]:
        arr = np.ones((2, anchor.height, anchor.width), dtype="uint16")
        tile = anchor.with_np(arr, ["B02", "B03"]).with_metadata({"description": "toy image"})
        return {"image": tile}

    def context(self, tiles: dict[str, GeoTile]) -> dict:
        ref = next(iter(tiles.values()))
        return {"datetime": ref.start.isoformat()}


def _toy_anchor() -> GeoAnchor:
    return GeoAnchor(
        geobox=GeoBox.from_bbox((500000, 5000000, 500320, 5000320), crs=UTM, resolution=10, anchor="edge"),
        datetime=datetime(2023, 2, 1),
    )


class TestStreamIngest:
    def test_renders_sample(self):
        anchor = _toy_anchor()
        samples = list(
            stream_ingest(
                _ToyPipeline(),
                [anchor],
                sel_bands={"image": ["B02"]},
                dtype_override={"image": torch.float32},
            )
        )

        assert len(samples) == 1
        assert tuple(samples[0]["image"].shape) == (1, 32, 32)
        assert samples[0]["image"].dtype == torch.float32
        assert samples[0]["context"]["datetime"] == "2023-02-01T00:00:00"

    def test_no_files_written(self, tmp_path):
        """stream_ingest never touches disk — nothing to assert against tmp_path except that it stays empty."""
        list(stream_ingest(_ToyPipeline(), [_toy_anchor()]))
        assert list(tmp_path.iterdir()) == []


class TestSaveDataset:
    def test_writes_one_anchor_folder(self, tmp_path):
        anchor = _toy_anchor()
        save_dataset(_ToyPipeline(), [anchor], tmp_path)
        assert (tmp_path / f"{anchor.stem}{GEOSTACK_SUFFIX}" / "image.zarr").exists()

    def test_accepts_plain_list_of_geotile(self, tmp_path):
        """save_dataset takes any Iterable[GeoTile] — no AnchorSource required."""
        anchors = [_toy_anchor()]
        save_dataset(_ToyPipeline(), anchors, tmp_path)
        assert (tmp_path / f"{anchors[0].stem}{GEOSTACK_SUFFIX}" / "image.zarr").exists()

    def test_resumable_skips_processed_anchor(self, tmp_path):
        anchor = _toy_anchor()
        save_dataset(_ToyPipeline(), [anchor], tmp_path)

        call_count = 0
        original_ingest = _ToyPipeline.ingest

        def counting_ingest(self, anchor):
            nonlocal call_count
            call_count += 1
            return original_ingest(self, anchor)

        _ToyPipeline.ingest = counting_ingest
        try:
            save_dataset(_ToyPipeline(), [anchor], tmp_path)
        finally:
            _ToyPipeline.ingest = original_ingest

        assert call_count == 0  # already marked done, re-ingest skipped

    def test_manifest_dir_not_picked_up_as_anchor(self, tmp_path):
        anchor = _toy_anchor()
        save_dataset(_ToyPipeline(), [anchor], tmp_path)
        ds = GeoDataset(tmp_path)
        assert len(ds) == 1  # the "toy" manifest-tracking dir isn't a sample

    def test_manifest_records_pipeline_and_layer_metadata(self, tmp_path):
        save_dataset(_ToyPipeline(), [_toy_anchor()], tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["metadata"]["pipeline"] == "_ToyPipeline"
        assert manifest["metadata"]["layers"]["image"]["description"] == "toy image"

    def test_manifest_metadata_untouched_on_fully_resumed_run(self, tmp_path):
        """A run that skips every anchor (already done) doesn't overwrite metadata with {}."""
        anchor = _toy_anchor()
        save_dataset(_ToyPipeline(), [anchor], tmp_path)
        save_dataset(_ToyPipeline(), [anchor], tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["metadata"]["layers"]["image"]["description"] == "toy image"
