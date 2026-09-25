"""Verify the artifact and remote-loading branch used by the Hub mixin."""

import json
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import ModelCard
import torch

from geosave_engine.ml.models.contract import ModelChain


def test_push_uploads_complete_reconstructable_artifact(pushed_artifact: Path) -> None:
    assert {path.name for path in pushed_artifact.iterdir()} == {
        "README.md",
        "config.json",
        "model.safetensors",
    }
    config = json.loads((pushed_artifact / "config.json").read_text())
    assert list(config["stages"]) == ["z_encoder", "a_head"]
    assert config["stages"]["z_encoder"]["init_args"]["pretrained"] is False
    card = ModelCard.load(pushed_artifact / "README.md")
    assert card.data.to_dict()["library_name"] == "geosave-engine"
    restored = ModelChain.from_pretrained(pushed_artifact, local_files_only=True)
    torch.testing.assert_close(restored(torch.tensor(3.0)), torch.tensor(23.0))


def test_remote_loader_reads_uploaded_config_and_weights(pushed_artifact: Path) -> None:
    requests: list[str] = []

    def download(*, repo_id: str, filename: str, **kwargs: object) -> str:
        assert repo_id == "test-org/geosave-probe"
        assert kwargs["revision"] == "test-commit"
        requests.append(filename)
        return str(pushed_artifact / filename)

    with patch("huggingface_hub.hub_mixin.hf_hub_download", side_effect=download):
        restored = ModelChain.from_pretrained(
            "test-org/geosave-probe", revision="test-commit"
        )
    assert requests == ["config.json", "model.safetensors"]
    assert not restored.training
    torch.testing.assert_close(restored(torch.tensor(3.0)), torch.tensor(23.0))
