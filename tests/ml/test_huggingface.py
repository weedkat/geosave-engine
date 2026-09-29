"""Transformers is the sole persistence adapter for configured model chains."""

from __future__ import annotations

import json
import os
from pathlib import Path
from shutil import copy2
from types import SimpleNamespace
import subprocess
import sys
from unittest.mock import MagicMock, patch

from huggingface_hub import ModelCard
import pytest
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModel

from geosave_engine.__about__ import __version__
from geosave_engine.ml.huggingface import GeoSaveConfig, GeoSaveModel
from geosave_engine.ml.model_chain import ModelChain
from geosave_engine.ml.models.head.dense import DenseHead
from geosave_engine.ml.registry import build_model
from tests.ml.model_chain.conftest import Head


@pytest.fixture
def stages() -> dict[str, dict[str, object]]:
    return {
        "head": {
            "name": "dense",
            "init_args": {"feature_channels": 2, "num_classes": 3},
        }
    }


def test_transformers_roundtrip_preserves_recipe_weights_outputs_and_mode(
    tmp_path: Path,
    stages: dict[str, dict[str, object]],
) -> None:
    chain = build_model(stages)
    head = chain.get_submodule("head")
    assert isinstance(head, DenseHead)
    with torch.no_grad():
        head.layers[-1].weight.fill_(2.0)
        head.layers[-1].bias.fill_(1.0)
    published = GeoSaveModel.from_chain(chain)
    published.eval()
    inputs = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)

    published.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    restored = AutoModel.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(restored, GeoSaveModel)
    assert not restored.training
    assert list(restored.chain.stage_specs) == ["head"]
    assert config["format_version"] == 1
    assert config["geosave_version"] == __version__
    assert "auto_map" not in config
    assert not (tmp_path / "huggingface.py").exists()
    torch.testing.assert_close(restored(feature_map=inputs), published(feature_map=inputs))
    for name, value in chain.state_dict().items():
        torch.testing.assert_close(restored.chain.state_dict()[name], value)


def test_export_rejects_a_directly_composed_chain() -> None:
    chain = ModelChain(head=DenseHead(feature_channels=2, num_classes=3))

    with pytest.raises(ValueError, match="stage specifications"):
        GeoSaveModel.from_chain(chain)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "Missing keys"),
        ("unexpected", "Unexpected keys"),
        ("mismatched", "Mismatched keys|size mismatch"),
    ],
)
def test_transformers_reload_rejects_incompatible_weights(
    tmp_path: Path,
    stages: dict[str, dict[str, object]],
    change: str,
    message: str,
) -> None:
    GeoSaveModel.from_chain(build_model(stages)).save_pretrained(tmp_path)
    weights_path = tmp_path / "model.safetensors"
    state = load_file(weights_path)
    if change == "missing":
        state.pop(next(iter(state)))
    elif change == "unexpected":
        state["unexpected"] = torch.tensor(1.0)
    else:
        key = next(iter(state))
        state[key] = torch.zeros(1)
    save_file(state, weights_path)

    with pytest.raises(RuntimeError, match=message):
        AutoModel.from_pretrained(tmp_path, local_files_only=True)


@pytest.mark.slow
def test_installed_adapter_loads_in_fresh_process(
    tmp_path: Path,
) -> None:
    chain = build_model(
        {
            "head": {
                "name": "dense",
                "init_args": {"feature_channels": 2, "num_classes": 3},
            }
        }
    )
    published = GeoSaveModel.from_chain(chain)
    published.eval()
    inputs = torch.ones(1, 2, 4, 4)
    torch.save(published(feature_map=inputs).detach(), tmp_path / "expected.pt")
    published.save_pretrained(tmp_path / "export")
    config = json.loads((tmp_path / "export" / "config.json").read_text())
    assert "auto_map" not in config
    assert not (tmp_path / "export" / "huggingface.py").exists()

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
from geosave_engine.ml.huggingface import GeoSaveModel
model = GeoSaveModel.from_pretrained(sys.argv[1], local_files_only=True)
actual = model(feature_map=torch.ones(1, 2, 4, 4))
expected = torch.load(sys.argv[2], weights_only=True)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
""",
            str(tmp_path / "export"),
            str(tmp_path / "expected.pt"),
        ],
        env={
            **os.environ,
            "HF_MODULES_CACHE": str(tmp_path / "modules"),
            "HF_HUB_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_publication_rejects_class_path_stages() -> None:
    chain = build_model(
        {
            "head": {
                "class_path": "tests.ml.model_chain.conftest.Head",
                "init_args": {"channels": 2},
            }
        }
    )

    with pytest.raises(ValueError, match="registered name"):
        GeoSaveModel.from_chain(chain)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", 99, "format version"),
        ("geosave_version", "0.0.0", "GeoSave version"),
    ],
)
def test_loading_rejects_incompatible_artifact_metadata(
    tmp_path: Path,
    stages: dict[str, dict[str, object]],
    field: str,
    value: object,
    message: str,
) -> None:
    GeoSaveModel.from_chain(build_model(stages)).save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))

    with pytest.raises((ValueError, RuntimeError), match=message):
        GeoSaveModel.from_pretrained(tmp_path, local_files_only=True)


def test_config_rejects_duplicate_stage_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        GeoSaveConfig(
            stages=[
                {"stage": "encoder", "spec": {"name": "dense"}},
                {"stage": "encoder", "spec": {"name": "dense"}},
            ]
        )


def test_adapter_push_uploads_a_reloadable_transformers_artifact(
    tmp_path: Path,
    stages: dict[str, dict[str, object]],
) -> None:
    destination = tmp_path / "uploaded"
    destination.mkdir()
    adapter = GeoSaveModel.from_chain(build_model(stages))
    api = MagicMock()
    api.create_repo.return_value = SimpleNamespace(repo_id="test-org/geosave-probe")

    def create_commit(*, operations: list[object], **kwargs: object) -> str:
        assert kwargs["repo_id"] == "test-org/geosave-probe"
        for operation in operations:
            copy2(operation.path_or_fileobj, destination / operation.path_in_repo)
        return "https://huggingface.co/test-org/geosave-probe/commit/test"

    api.create_commit.side_effect = create_commit
    with (
        patch("transformers.utils.hub.hf_api", return_value=api),
        patch(
            "transformers.utils.hub.create_and_tag_model_card",
            return_value=ModelCard(""),
        ),
    ):
        result = adapter.push_to_hub(
            "test-org/geosave-probe",
            token="test-token",
            revision="test-branch",
            create_pr=True,
            commit_message="Test model publication",
        )

    assert result.endswith("/commit/test")
    assert {path.name for path in destination.iterdir()} >= {
        "README.md",
        "config.json",
        "model.safetensors",
    }
    restored = AutoModel.from_pretrained(destination, local_files_only=True)
    inputs = torch.ones(1, 2, 3, 3)
    torch.testing.assert_close(
        restored(feature_map=inputs), adapter(feature_map=inputs)
    )
