import pytest
import torch

from geosave_engine.model.encoder import clay


def test_local_checkpoint_needs_no_published_model_source(tmp_path, monkeypatch):
    # A small real module isolates checkpoint selection from backbone construction.
    monkeypatch.setattr(clay, "Encoder", lambda **kwargs: torch.nn.Linear(2, 2))

    def download(**kwargs):
        pytest.fail("an explicit checkpoint must not access the Hub")

    monkeypatch.setattr(clay, "hf_hub_download", download)
    expected = torch.nn.Linear(2, 2)
    path = tmp_path / "clay.ckpt"
    torch.save(
        {
            "state_dict": {
                **{
                    f"model.encoder.{name}": value
                    for name, value in expected.state_dict().items()
                },
                "model.decoder.unused": torch.zeros(1),
            }
        },
        path,
    )

    loaded = clay.build_clay(
        "clay_v15_tiny", pretrained=True, checkpoint_path=str(path)
    )
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_unpublished_variant_without_local_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="no published checkpoint"):
        clay.build_clay("clay_v15_tiny", pretrained=True)
