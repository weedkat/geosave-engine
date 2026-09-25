"""Local Hugging Face artifacts reconstruct complete model chains."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from geosave_engine.ml.models.contract import ModelChain, Published, chain_step
from geosave_engine.ml.registry import StageSpec


class Encoder(nn.Module):
    channels: Published[int]

    def __init__(self, channels: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.pretrained = pretrained
        self.scale = nn.Parameter(torch.tensor(2.0 if pretrained else 0.0))

    @chain_step(outputs=("features",))
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.scale


class Head(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    @chain_step(head=True)
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.channels


def stages() -> dict[str, StageSpec]:
    return {
        "z_encoder": {"class_path": f"{__name__}.Encoder"},
        "a_head": {"class_path": f"{__name__}.Head"},
    }


def test_hub_roundtrip_preserves_weights_order_and_constructor_arguments(tmp_path):
    specs = stages()
    original = deepcopy(specs)
    model = ModelChain(stages=specs)
    encoder = model.get_submodule("z_encoder")
    assert isinstance(encoder, Encoder)
    assert encoder.pretrained
    with torch.no_grad():
        encoder.scale.fill_(7.0)
    model.save_pretrained(tmp_path)
    specs["z_encoder"]["init_args"] = {"channels": 99}

    config = json.loads((tmp_path / "config.json").read_text())
    assert list(config["stages"]) == ["z_encoder", "a_head"]
    assert config["stages"]["z_encoder"]["init_args"] == {
        "channels": 2,
        "pretrained": False,
    }
    assert config["stages"]["a_head"]["init_args"] == {"channels": 2}
    assert original == stages()
    assert (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / "README.md").is_file()

    restored = ModelChain.from_pretrained(tmp_path, local_files_only=True)
    assert not restored.training
    restored_encoder = restored.get_submodule("z_encoder")
    assert isinstance(restored_encoder, Encoder)
    assert not restored_encoder.pretrained
    assert list(restored._modules) == ["z_encoder", "a_head"]
    torch.testing.assert_close(restored(torch.tensor(3.0)), torch.tensor(23.0))
    assert restored.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_hub_artifact_loads_in_fresh_offline_process(tmp_path):
    model = ModelChain(stages=stages())
    model.save_pretrained(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import socket, sys, torch\n"
            "def no_network(*args, **kwargs):\n"
            "    raise AssertionError('Model reload attempted network access')\n"
            "socket.socket.connect = no_network\n"
            "from geosave_engine.ml.models.contract import ModelChain\n"
            "model = ModelChain.from_pretrained(sys.argv[1], local_files_only=True)\n"
            "assert model.z_encoder.pretrained is False\n"
            "torch.testing.assert_close(model(torch.tensor(3.0)), torch.tensor(8.0))\n",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hub_loading_rejects_unexpected_weights_by_default(tmp_path):
    model = ModelChain(stages=stages())
    model.register_parameter("unexpected", nn.Parameter(torch.tensor(1.0)))
    model.save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="unexpected"):
        ModelChain.from_pretrained(tmp_path, local_files_only=True)


def test_module_instances_require_a_construction_recipe_before_export(tmp_path):
    model = ModelChain(encoder=Encoder(), head=Head(2))
    with pytest.raises(ValueError, match="stage specifications"):
        model.save_pretrained(tmp_path)
    assert not (tmp_path / "model.safetensors").exists()


def test_stages_and_module_instances_cannot_be_mixed():
    with pytest.raises(ValueError, match="not both"):
        ModelChain(Encoder(), stages=stages())
