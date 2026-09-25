"""Optional Transformers publication preserves model construction and weights."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from geosave_engine.ml.registry import build_model
from geosave_engine.ml.models.contract import ModelChain

transformers = pytest.importorskip("transformers")


from geosave_engine.ml.huggingface import GeoSaveConfig, GeoSaveModel  # noqa: E402


def test_plain_mixin_artifact_is_not_an_automodel(pushed_artifact):
    with pytest.raises(ValueError, match="Unrecognized model"):
        transformers.AutoModel.from_pretrained(pushed_artifact, local_files_only=True)


def test_registered_transformers_model_loads_identical_outputs(
    tmp_path, pushed_artifact
):
    original = ModelChain.from_pretrained(pushed_artifact, local_files_only=True)
    wrapper = GeoSaveModel.from_chain(original)
    assert wrapper.chain is original
    wrapper.save_pretrained(tmp_path / "transformers")

    restored = transformers.AutoModel.from_pretrained(
        tmp_path / "transformers", local_files_only=True, trust_remote_code=False
    )

    assert isinstance(restored, GeoSaveModel)
    assert list(restored.chain._modules) == ["z_encoder", "a_head"]
    torch.testing.assert_close(restored(image=torch.tensor(3.0)), torch.tensor(23.0))
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.chain.state_dict()[key], value)


def test_exported_adapter_loads_without_registration_in_fresh_process(tmp_path):
    model = build_model(
        {
            "head": {
                "name": "dense",
                "init_args": {"feature_channels": 2, "num_classes": 3},
            },
        }
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    wrapper = GeoSaveModel.from_chain(model)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    wrapper.eval()
    inputs = torch.ones(1, 2, 4, 4)
    torch.save(wrapper(feature_map=inputs).detach(), tmp_path / "expected.pt")
    wrapper.save_pretrained(tmp_path / "export")
    config = json.loads((tmp_path / "export" / "config.json").read_text())
    assert set(config["auto_map"]) == {"AutoConfig", "AutoModel"}
    assert (tmp_path / "export" / "huggingface.py").is_file()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained(
    sys.argv[1], trust_remote_code=True, local_files_only=True,
)
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


def test_export_requires_recipe_and_rejects_duplicate_stages():
    from tests.ml.models.contract.test_hub import Encoder

    with pytest.raises(ValueError, match="Build with stages"):
        GeoSaveModel.from_chain(ModelChain(encoder=Encoder()))
    with pytest.raises(ValueError, match="unique"):
        GeoSaveConfig(
            stages=[
                {"stage": "encoder", "spec": {"name": "dense"}},
                {"stage": "encoder", "spec": {"name": "dense"}},
            ]
        )


def test_wrapping_preserves_trained_parameters_and_mode(pushed_artifact):
    chain = ModelChain.from_pretrained(pushed_artifact)
    before = {name: value.clone() for name, value in chain.state_dict().items()}
    wrapper = GeoSaveModel.from_chain(chain)
    assert not wrapper.training
    assert wrapper.chain is chain
    for name, value in before.items():
        torch.testing.assert_close(
            wrapper.chain.state_dict()[name], value, rtol=0, atol=0
        )
    specs = chain.stage_specs
    specs.clear()
    assert list(chain.stage_specs) == ["z_encoder", "a_head"]
