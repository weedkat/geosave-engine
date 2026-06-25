from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.eva import Eva

from geosave_engine.ml.models.decoder.dpt import DPTDecoder
from geosave_engine.ml.models.encoder.dinov3 import TIMM_MODELS, build_dinov3_timm
from geosave_engine.ml.models.head.dpt import DPTRegHead, DPTSegHead


class DPTDinoV3(nn.Module):
    """
    Dense Prediction Transformer (DPT) using a DINOv3/EVA backbone.

    This model extracts multi-scale intermediate features from a ViT, fuses them using 
    the DPT decoder, applies a task-specific head, and upsamples the prediction back 
    to the original input resolution.
    """

    def __init__(
        self,
        # --- backbone ---
        backbone: str,
        pretrained: bool = True,
        in_channels: int = 3,
        img_size: int | tuple[int, int] = 224,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_values: float | None = 1e-5,
        num_reg_tokens: int = 4,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        # --- decoder ---
        readout: Literal['cat', 'add', 'ignore'] = 'cat',
        intermediate_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024),
        fusion_channels: int = 256,
        # --- head ---
        task: Literal['segmentation', 'regression'] = 'segmentation',
        num_classes: int = 1,
        head_dropout: float = 0.0,
    ):
        super().__init__()

        # --- 1. Backbone (Encoder) ---
        self.encoder: Eva = build_dinov3_timm(
            backbone,
            pretrained=pretrained,
            in_channels=in_channels,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_values=init_values,
            num_reg_tokens=num_reg_tokens,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        
        self.out_indices: list[int] = list(TIMM_MODELS[backbone]['out_indices'])
        
        # Extract channel dimensions and spatial strides dynamically from the timm model
        feature_info: list = self.encoder.feature_info
        encoder_out_channels = [int(feature_info[i]['num_chs']) for i in self.out_indices]
        encoder_output_strides = [int(feature_info[i]['reduction']) for i in self.out_indices]

        # --- 2. Decoder ---
        self.decoder = DPTDecoder(
            encoder_out_channels=encoder_out_channels,
            encoder_output_strides=encoder_output_strides,
            readout=readout,
            intermediate_channels=intermediate_channels,
            fusion_channels=fusion_channels,
        )

        # --- 3. Head ---
        if task == 'segmentation':
            self.head: nn.Module = DPTSegHead(
                fusion_channels, num_classes=num_classes, dropout=head_dropout,
            )
        elif task == 'regression':
            self.head = DPTRegHead(
                fusion_channels, num_outputs=num_classes, dropout=head_dropout,
            )
        else:
            raise ValueError(f"task must be 'segmentation' or 'regression', got {task!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - The original input image.

        Returns:
            y: (B, num_classes, H, W) - The final dense prediction map.
        """
        # Save original dimensions for the final upsampling step
        _, _, H_in, W_in = x.shape

        # --- Step 1: Encoder Extraction ---
        # Returns a list of tuples containing (features, prefix_tokens) for each specified layer.
        # output_fmt='NCHW' handles the unflattening from (B, N, C) to (B, C, H/stride, W/stride) internally.
        pairs = self.encoder.forward_intermediates(
            x,
            indices=self.out_indices,
            intermediates_only=True,
            return_prefix_tokens=True,
            output_fmt='NCHW',
        )
        
        # Unzip pairs into two separate lists:
        # features: List of 4 tensors of shape (B, C_i, H/stride, W/stride)
        # prefix_tokens: List of 4 tensors of shape (B, num_prefix, C_i)
        features, prefix_tokens = map(list, zip(*pairs))

        # --- Step 2: DPT Decoder Fusion ---
        # Reassembles the identically sized ViT features into a multi-scale pyramid and fuses them.
        # Output is typically at 1/2 or 1/4 of the original resolution.
        # (B, C_i, H/stride, W/stride) -> (B, fusion_channels, H_fused, W_fused)
        fused = self.decoder(features, prefix_tokens)

        # --- Step 3: Task Head ---
        # Compresses the fusion channels down to the target number of classes/outputs.
        # (B, fusion_channels, H_fused, W_fused) -> (B, num_classes, H_fused, W_fused)
        y = self.head(fused)

        # --- Step 4: Final Bilinear Upsampling ---
        # Restores the prediction map to the exact (H, W) of the original image.
        # (B, num_classes, H_fused, W_fused) -> (B, num_classes, H_in, W_in)
        if y.shape[-2:] != (H_in, W_in):
            y = F.interpolate(y, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return y