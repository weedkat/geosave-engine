"""Native model release persistence and independent loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from geosave_engine.__about__ import __version__
from geosave_engine.ml.model_chain import ModelChain
from geosave_engine.ml.registry import build_model
from geosave_engine.release import load_model, load_spec, save_model
from geosave_engine.workflow.specs import ModelSpec, Ref


@pytest.fixture
def model() -> ModelChain:
    return build_model(
        {
            "head": {
                "name": "dense",
                "init_args": {"feature_channels": 2, "num_classes": 3},
            }
        }
    )


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {},
            "model_inputs": {
                "feature_map": {"call": Ref("prepared_feature_map")}
            },
        }
    )


def test_local_release_contains_only_complete_inference_artifact(
    tmp_path: Path,
    model: ModelChain,
    spec: ModelSpec,
) -> None:
    target = save_model(model, tmp_path / "release", spec=spec)

    assert target == tmp_path / "release"
    assert {path.name for path in target.iterdir()} == {
        "README.md",
        "config.json",
        "model.safetensors",
        "model_spec.yaml",
    }
    assert f"geosave-engine[hub]=={__version__}" in (
        target / "README.md"
    ).read_text()
    config = json.loads((target / "config.json").read_text())
    assert config["stages"] == [
        {
            "stage": "head",
            "spec": {
                "name": "dense",
                "init_args": {
                    "dropout": 0.0,
                    "feature_channels": 2,
                    "hidden_channels": None,
                    "input_size": None,
                    "num_classes": 3,
                },
            },
        }
    ]


def test_local_model_and_spec_load_independently(
    tmp_path: Path,
    model: ModelChain,
    spec: ModelSpec,
) -> None:
    model.eval()
    inputs = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    target = save_model(model, tmp_path / "release", spec=spec)

    restored = load_model(target)
    restored_spec = load_spec(target)

    assert isinstance(restored, ModelChain)
    assert restored_spec == spec
    torch.testing.assert_close(restored(feature_map=inputs), model(feature_map=inputs))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_existing_destination_is_not_modified(
    tmp_path: Path,
    model: ModelChain,
    spec: ModelSpec,
) -> None:
    target = tmp_path / "release"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError):
        save_model(model, target, spec=spec)

    assert marker.read_text() == "keep"


def test_missing_local_path_is_not_treated_as_a_hub_repository(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"

    with patch("geosave_engine.release.hf_hub_download") as download:
        with pytest.raises(FileNotFoundError):
            load_spec(missing)
        with pytest.raises(FileNotFoundError):
            load_model(missing)

    download.assert_not_called()


def test_missing_model_input_aborts_before_creating_destination(
    tmp_path: Path,
    model: ModelChain,
) -> None:
    target = tmp_path / "release"
    incompatible = ModelSpec(schema_version=2, rasters={})

    with pytest.raises(ValueError, match="model inputs"):
        save_model(model, target, spec=incompatible)

    assert not target.exists()


def test_failed_save_leaves_no_release_or_staging_directory(
    tmp_path: Path,
    model: ModelChain,
    spec: ModelSpec,
) -> None:
    target = tmp_path / "release"

    with (
        patch(
            "geosave_engine.ml.huggingface.GeoSaveModel.save_pretrained",
            side_effect=RuntimeError("injected failure"),
        ),
        pytest.raises(RuntimeError, match="injected failure"),
    ):
        save_model(model, target, spec=spec)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_remote_spec_load_downloads_only_the_spec(tmp_path: Path) -> None:
    remote_spec = ModelSpec(schema_version=2, rasters={})
    path = remote_spec.save(tmp_path / "download")

    with patch(
        "geosave_engine.release.hf_hub_download", return_value=str(path)
    ) as download:
        actual = load_spec(
            "org/model",
            revision="abc",
            token="token",
            local_files_only=True,
        )

    assert actual == remote_spec
    download.assert_called_once_with(
        repo_id="org/model",
        filename=ModelSpec.filename,
        revision="abc",
        token="token",
        local_files_only=True,
    )


@pytest.mark.slow
def test_local_spec_load_does_not_import_transformers(tmp_path: Path) -> None:
    path = ModelSpec(schema_version=2, rasters={}).save(tmp_path / "release")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from geosave_engine.release import load_spec
load_spec(sys.argv[1])
assert "transformers" not in sys.modules
""",
            str(path.parent),
        ],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
