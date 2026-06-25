from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForDepthEstimation,
    AutoModelForSemanticSegmentation,
    DPTConfig,
    DPTImageProcessor,
)


class DPTHF(nn.Module):
    """DPT from a HuggingFace checkpoint, task-dispatched via AutoModelFor*.

    Forward output is normalized to a single Tensor (logits for segmentation,
    predicted_depth for depth) and bilinearly upsampled to input HxW.

    All `None` config overrides inherit the checkpoint's pretrained value.
    """

    def __init__(
        self,
        model_name: str,
        task: Literal['segmentation', 'depth'] = 'segmentation',
        # --- segmentation-only ---
        num_labels: int | None = None,
        semantic_loss_ignore_index: int | None = None,
        semantic_classifier_dropout: float | None = None,
        # --- backbone overrides ---
        num_channels: int | None = None,
        image_size: int | tuple[int, int] | None = None,
        hidden_dropout_prob: float | None = None,
        attention_probs_dropout_prob: float | None = None,
        # --- DPT neck/head overrides ---
        readout_type: Literal['ignore', 'add', 'project'] | None = None,
        backbone_out_indices: tuple[int, ...] | None = None,
        neck_hidden_sizes: tuple[int, ...] | None = None,
        fusion_hidden_size: int | None = None,
        use_auxiliary_head: bool | None = None,
        auxiliary_loss_weight: float | None = None,
        # --- loading ---
        ignore_mismatched_sizes: bool = True,
    ):
        super().__init__()

        if task not in ('segmentation', 'depth'):
            raise ValueError(f"task must be 'segmentation' or 'depth', got {task!r}")

        processor = DPTImageProcessor.from_pretrained(model_name)
        config = DPTConfig.from_pretrained(model_name)

        if task == 'segmentation' and num_labels is not None:
            config.num_labels = num_labels

        if image_size is None and isinstance(processor.size, dict):
            h: int = processor.size.get('height') or processor.size['shortest_edge']
            w: int = processor.size.get('width') or h
            image_size = (h, w) if h != w else h

        overrides: dict[str, object] = {
            'num_channels': num_channels,
            'image_size': image_size,
            'hidden_dropout_prob': hidden_dropout_prob,
            'attention_probs_dropout_prob': attention_probs_dropout_prob,
            'readout_type': readout_type,
            'backbone_out_indices': list(backbone_out_indices) if backbone_out_indices is not None else None,
            'neck_hidden_sizes': list(neck_hidden_sizes) if neck_hidden_sizes is not None else None,
            'fusion_hidden_size': fusion_hidden_size,
            'use_auxiliary_head': use_auxiliary_head,
            'auxiliary_loss_weight': auxiliary_loss_weight,
            'semantic_loss_ignore_index': semantic_loss_ignore_index,
            'semantic_classifier_dropout': semantic_classifier_dropout,
        }
        for attr, val in overrides.items():
            if val is not None:
                setattr(config, attr, val)

        if task == 'segmentation' and num_labels is None:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name, config=config, ignore_mismatched_sizes=ignore_mismatched_sizes)
            self._out_key = 'logits'
        elif task == 'depth':
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name, config=config, ignore_mismatched_sizes=ignore_mismatched_sizes)
            self._out_key = 'predicted_depth'

        self.img_mean: list[float] = list(processor.image_mean)
        self.img_std: list[float] = list(processor.image_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        y: torch.Tensor = getattr(out, self._out_key)
        if y.ndim == 3:
            y = y.unsqueeze(1)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return y
