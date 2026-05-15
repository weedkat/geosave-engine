from pathlib import Path

import pytest

from templates.semantic_segmentation.methods.supervised.src.data_module import (
    GeosaveDataModule,
    _normalize_split_paths,
)


class TestNormalizeSplitPaths:
    def test_string_becomes_single_element_list(self):
        assert _normalize_split_paths({"train": "/a"}) == {"train": [Path("/a")]}

    def test_list_preserved_as_path_list(self):
        assert _normalize_split_paths({"train": ["/a", "/b"]}) == {
            "train": [Path("/a"), Path("/b")]
        }

    def test_multiple_splits_all_converted(self):
        result = _normalize_split_paths({"train": "/a", "val": "/b", "test": "/c"})
        assert set(result.keys()) == {"train", "val", "test"}
        assert all(isinstance(v, list) for v in result.values())

    def test_empty_dict_returns_empty(self):
        assert _normalize_split_paths({}) == {}

    def test_single_item_list_stays_list(self):
        result = _normalize_split_paths({"train": ["/a"]})
        assert result == {"train": [Path("/a")]}

    def test_values_are_path_objects(self):
        result = _normalize_split_paths({"train": "/data/train"})
        assert all(isinstance(p, Path) for p in result["train"])


class TestGeosaveDataModuleInit:
    def test_image_paths_normalized(self):
        dm = GeosaveDataModule(data_image={"train": "/img/train"})
        assert dm.data_image == {"train": [Path("/img/train")]}

    def test_label_paths_normalized_when_given(self):
        dm = GeosaveDataModule(
            data_image={"train": "/img"},
            data_label={"train": "/lbl"},
        )
        assert dm.data_label == {"train": [Path("/lbl")]}

    def test_mask_paths_normalized_when_given(self):
        dm = GeosaveDataModule(
            data_image={"train": "/img"},
            data_mask={"train": "/msk"},
        )
        assert dm.data_mask == {"train": [Path("/msk")]}

    def test_label_none_by_default(self):
        assert GeosaveDataModule(data_image={"train": "/img"}).data_label is None

    def test_mask_none_by_default(self):
        assert GeosaveDataModule(data_image={"train": "/img"}).data_mask is None

    def test_mean_without_std_raises(self):
        with pytest.raises(ValueError, match="mean_norm and std_norm"):
            GeosaveDataModule(data_image={"train": "/img"}, mean_norm=[0.5])

    def test_std_without_mean_raises(self):
        with pytest.raises(ValueError, match="mean_norm and std_norm"):
            GeosaveDataModule(data_image={"train": "/img"}, std_norm=[0.5])

    def test_both_norm_none_accepted(self):
        dm = GeosaveDataModule(data_image={"train": "/img"})
        assert dm.mean_norm is None and dm.std_norm is None

    def test_both_norm_set_accepted(self):
        dm = GeosaveDataModule(
            data_image={"train": "/img"},
            mean_norm=[0.5, 0.5],
            std_norm=[0.2, 0.2],
        )
        assert dm.mean_norm == [0.5, 0.5]
        assert dm.std_norm == [0.2, 0.2]

    def test_predict_stride_defaults_to_patch_size(self):
        dm = GeosaveDataModule(data_image={"train": "/img"}, predict_patch_size=512)
        assert dm.predict_stride == 512

    def test_predict_stride_explicit_overrides_default(self):
        dm = GeosaveDataModule(
            data_image={"train": "/img"},
            predict_patch_size=512,
            predict_stride=256,
        )
        assert dm.predict_stride == 256

    def test_batch_size_stored(self):
        dm = GeosaveDataModule(data_image={"train": "/img"}, batch_size=32)
        assert dm.batch_size == 32

    def test_num_workers_stored(self):
        dm = GeosaveDataModule(data_image={"train": "/img"}, num_workers=4)
        assert dm.num_workers == 4


class TestGeosaveDataModuleSetup:
    def test_invalid_stage_raises(self):
        dm = GeosaveDataModule(data_image={"train": "/img", "val": "/img"})
        with pytest.raises(ValueError, match="Invalid stage"):
            dm.setup("bad_stage")

    def test_none_stage_raises(self):
        dm = GeosaveDataModule(data_image={"train": "/img"})
        with pytest.raises(ValueError):
            dm.setup(None)


class TestBuildDataset:
    def test_missing_split_raises(self):
        dm = GeosaveDataModule(data_image={"train": "/img/train"})
        with pytest.raises(ValueError, match="No image paths configured for split 'val'"):
            dm._build_dataset("val")

    def test_empty_image_dict_raises_for_any_split(self):
        dm = GeosaveDataModule(data_image={})
        with pytest.raises(ValueError, match="No image paths configured"):
            dm._build_dataset("train")
