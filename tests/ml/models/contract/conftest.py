"""Exercise real Hub serialization while replacing only the network API."""

from pathlib import Path
from shutil import copytree
from unittest.mock import patch

from huggingface_hub.hf_api import RepoUrl
import pytest
import torch

from geosave_engine.ml.models.contract import ModelChain
from tests.ml.models.contract.test_hub import Encoder, stages


@pytest.fixture
def pushed_artifact(tmp_path: Path) -> Path:
    """Capture exactly the folder passed by ModelChain.push_to_hub to the API."""
    destination = tmp_path / "uploaded"
    repo_id = "test-org/geosave-probe"
    commit_url = f"https://huggingface.co/{repo_id}/commit/test-commit"
    model = ModelChain(stages=stages())
    encoder = model.get_submodule("z_encoder")
    assert isinstance(encoder, Encoder)
    with torch.no_grad():
        encoder.scale.fill_(7.0)

    def upload(*, folder_path: Path, **kwargs: object) -> str:
        copytree(folder_path, destination)
        assert kwargs["repo_id"] == repo_id
        assert kwargs["repo_type"] == "model"
        assert kwargs["revision"] == "test-branch"
        assert kwargs["commit_message"] == "Test model publication"
        assert kwargs["create_pr"] is True
        return commit_url

    with patch("huggingface_hub.hub_mixin.HfApi", autospec=True) as api_class:
        api = api_class.return_value
        api.create_repo.return_value = RepoUrl(f"https://huggingface.co/{repo_id}")
        api.upload_folder.side_effect = upload
        result = model.push_to_hub(
            repo_id,
            private=True,
            token="test-token",
            branch="test-branch",
            create_pr=True,
            commit_message="Test model publication",
        )
        api_class.assert_called_once_with(token="test-token")
        api.create_repo.assert_called_once_with(
            repo_id=repo_id, private=True, exist_ok=True
        )
        api.upload_folder.assert_called_once()
        assert result == commit_url
    return destination
