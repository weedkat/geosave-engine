from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from lightning.pytorch import LightningModule
from terratorch.cli_tools import build_lightning_cli
from terratorch.models.model import ModelOutput
from terratorch.utils import remove_unexpected_prefix


_HF_REPO_ID: str = 'ibm-granite/granite-geospatial-biomass'
_CKPT_FILENAME: str = 'biomass_model.ckpt'
_CONFIG_FILENAME: str = 'config.yaml'


class GraniteAGB(nn.Module):
    """Granite geospatial biomass model with Terratorch's Lightning shell removed.

    The wrapped ``model`` is the trainable ``nn.Module`` built from the HuggingFace
    Terratorch config. Forward returns the primary biomass prediction tensor instead
    of Terratorch's ``ModelOutput`` container.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        output = self.model(x, **kwargs)
        if isinstance(output, ModelOutput):
            return output.output
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, 'output'):
            return cast(torch.Tensor, output.output)
        raise TypeError(
            f'expected Tensor or ModelOutput from Granite AGB model, got {type(output)!r}'
        )


def get_granite_agb_paths(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    repo_id: str = _HF_REPO_ID,
) -> tuple[Path, Path]:
    """Resolve local Granite AGB checkpoint and config paths.

    Args:
        checkpoint_path: Existing checkpoint path. ``None`` downloads from HF Hub.
        config_path: Existing Terratorch config path. ``None`` downloads from HF Hub.
        repo_id: HuggingFace repository id.

    Returns:
        ``(checkpoint_path, config_path)`` as local paths.
    """
    ckpt = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(hf_hub_download(repo_id=repo_id, filename=_CKPT_FILENAME))
    )
    config = (
        Path(config_path)
        if config_path is not None
        else Path(hf_hub_download(repo_id=repo_id, filename=_CONFIG_FILENAME))
    )
    return ckpt, config


def _strip_task_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split('.')
        if parts[0] == 'model':
            key = '.'.join(parts[1:])
        stripped[key] = value
    return stripped


def load_granite_agb_task(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    repo_id: str = _HF_REPO_ID,
    predict_dataset_bands: list[str] | None = None,
    predict_output_bands: list[str] | None = None,
) -> LightningModule:
    """Build the original Terratorch Lightning task for training/fine-tuning.

    Args:
        checkpoint_path: Existing checkpoint path. ``None`` downloads from HF Hub.
        config_path: Existing Terratorch config path. ``None`` downloads from HF Hub.
        repo_id: HuggingFace repository id.
        predict_dataset_bands: Optional band override passed through Terratorch CLI.
        predict_output_bands: Optional output-band override passed through Terratorch CLI.

    Returns:
        Terratorch Lightning task with checkpoint weights loaded into ``task.model``.
    """
    ckpt, config = get_granite_agb_paths(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        repo_id=repo_id,
    )
    arguments: list[object] = ['--config', config]

    if predict_dataset_bands is not None:
        arguments.extend([
            '--data.init_args.predict_dataset_bands',
            '[' + ','.join(predict_dataset_bands) + ']',
        ])
    if predict_output_bands is not None:
        arguments.extend([
            '--data.init_args.predict_output_bands',
            '[' + ','.join(predict_output_bands) + ']',
        ])

    cli = build_lightning_cli(arguments, run=False)
    task = cast(LightningModule, cli.model)

    weights = torch.load(ckpt, map_location='cpu', weights_only=True)
    if 'state_dict' in weights:
        weights = weights['state_dict']
    state_dict = _strip_task_prefix(remove_unexpected_prefix(weights))
    task.model.load_state_dict(state_dict)
    return task


def build_granite_agb(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    repo_id: str = _HF_REPO_ID,
    predict_dataset_bands: list[str] | None = None,
    predict_output_bands: list[str] | None = None,
) -> GraniteAGB:
    """Build the extracted Granite AGB ``nn.Module`` for direct use.

    Args:
        checkpoint_path: Existing checkpoint path. ``None`` downloads from HF Hub.
        config_path: Existing Terratorch config path. ``None`` downloads from HF Hub.
        repo_id: HuggingFace repository id.
        predict_dataset_bands: Optional band override passed through Terratorch CLI.
        predict_output_bands: Optional output-band override passed through Terratorch CLI.

    Returns:
        ``GraniteAGB`` wrapping the checkpoint-loaded Terratorch inner model.
    """
    task = load_granite_agb_task(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        repo_id=repo_id,
        predict_dataset_bands=predict_dataset_bands,
        predict_output_bands=predict_output_bands,
    )
    return GraniteAGB(task.model)
