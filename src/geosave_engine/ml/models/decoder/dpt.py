import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

from geosave_engine.ml.models.contract import model_context


class _ReadoutProjectBlock(nn.Module):
    """
    Concatenates the CLS token with each spatial position and projects it back to the original channel dimension.

    Args:
        embed_dim (int): The channel dimension of the input features.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor, prefix_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) - Spatial features from the encoder.
            prefix_tokens: (B, num_prefix, C) - Tokens including the CLS token.

        Returns:
            (B, C, H, W) - Features enriched with global CLS context.
        """
        B, C, H, W = features.shape
        
        # (B, C, H, W) -> (B, H * W, C)
        x = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Extract CLS token and expand: (B, num_prefix, C) -> (B, 1, C) -> (B, H * W, C)
        cls = prefix_tokens[:, :1].expand_as(x)
        
        # Concatenate and project: (B, H * W, 2C) -> (B, H * W, C)
        x = self.project(torch.cat([x, cls], dim=-1))
        
        # Reshape back to spatial dimensions: (B, H * W, C) -> (B, C, H, W)
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class _ReassembleBlock(nn.Module):
    """
    Projects encoder features to a target scale and uniform channel dimension.

    Args:
        in_channels (int): Channel dimension of the incoming feature map.
        mid_channels (int): Intermediate channel dimension.
        out_channels (int): Final output channel dimension for fusion.
        scale_factor (float): Factor to scale spatial resolution (e.g., 2.0 for upsample, 0.5 for downsample).
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, scale_factor: float):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        if scale_factor > 1.0:
            self.resample = nn.ConvTranspose2d(
                mid_channels, mid_channels, kernel_size=int(scale_factor), stride=int(scale_factor)
            )
        elif scale_factor == 1.0:
            self.resample = nn.Identity()
        else:
            self.resample = nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=int(1 / scale_factor), padding=1
            )

        self.proj_out = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) - Feature map from the encoder or readout block.

        Returns:
            (B, out_channels, H * scale_factor, W * scale_factor) - Rescaled and projected feature map.
        """
        # (B, in_channels, H, W) -> (B, mid_channels, H, W)
        x = self.proj_in(x)
        
        # (B, mid_channels, H, W) -> (B, mid_channels, H * scale_factor, W * scale_factor)
        x = self.resample(x)
        
        # (B, mid_channels, H_new, W_new) -> (B, out_channels, H_new, W_new)
        x = self.proj_out(x)
        
        return x


class _ResidualConvUnit(nn.Module):
    """
    Standard residual convolutional unit used within the fusion process.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H, W)
        """
        residual = x
        
        # (B, C, H, W) -> (B, C, H, W)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        
        # (B, C, H, W) + (B, C, H, W) -> (B, C, H, W)
        return x + residual


class _FusionBlock(nn.Module):
    """
    Fuses features from the current scale with features from the previous (coarser) scale.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.res1 = _ResidualConvUnit(channels)
        self.res2 = _ResidualConvUnit(channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        """
        Inputs:
            x: (B, C, H, W) - Feature map from the current scale.
            prev: (B, C, H, W) or None - Fused feature map from the previous coarser scale.

        Outputs:
            (B, C, 2H, 2W) - Fused and upsampled feature map.
        """
        # (B, C, H, W) -> (B, C, H, W)
        x = self.res1(x)
        
        if prev is not None:
            # (B, C, H, W) + (B, C, H, W) -> (B, C, H, W)
            x = x + prev
            
        # (B, C, H, W) -> (B, C, H, W)
        x = self.res2(x)
        
        # Bilinear upsample: (B, C, H, W) -> (B, C, 2H, 2W)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # (B, C, 2H, 2W) -> (B, C, 2H, 2W)
        x = self.proj(x)
        
        return x


class DPTDecoder(nn.Module):
    """
    Dense Prediction Transformer (DPT) Decoder.
    Takes multi-scale features from a Vision Transformer (ViT) and fuses them into a dense spatial representation.
    https://arxiv.org/pdf/2103.13413

    Args:
        encoder_out_channels: List of channel dimensions for each encoder output scale.
        encoder_output_strides: List of downsampling factors for each encoder output relative to the original image.
        readout: Method to integrate the CLS token ('cat', 'add', or 'ignore').
        intermediate_channels: Intermediate projection channels for each reassemble block.
        fusion_channels: The unified channel dimension used throughout the fusion process.
    """
    def __init__(
        self,
        encoder_out_channels: list[int],
        encoder_output_strides: list[int],
        readout: Literal['cat', 'add', 'ignore'] = 'cat',
        intermediate_channels: tuple[int, ...] = (256, 512, 1024, 1024),
        fusion_channels: int = 256,
    ):
        super().__init__()
        n = len(encoder_out_channels)
        
        self.readout = readout
        if readout == 'cat':
            self.readout_blocks = nn.ModuleList([
                _ReadoutProjectBlock(c) for c in encoder_out_channels
            ])

        # Calculate relative scale factors for reassembly targeting 1/4, 1/8, 1/16, 1/32 of original input
        scale_factors = [stride / (2 ** (i + 2)) for i, stride in enumerate(encoder_output_strides)]
        
        self.reassemble_blocks = nn.ModuleList([
            _ReassembleBlock(encoder_out_channels[i], intermediate_channels[i], fusion_channels, scale_factors[i])
            for i in range(n)
        ])
        
        self.fusion_blocks = nn.ModuleList([
            _FusionBlock(fusion_channels) for _ in range(n)
        ])

        self.out_channels: int = fusion_channels

    def forward(self, features: list[torch.Tensor], prefix_tokens: list[torch.Tensor | None]) -> torch.Tensor:
        """
        Args:
            features: List of (B, C_i, H_i, W_i) - Multi-scale spatial features from the encoder.
            prefix_tokens: List of (B, num_prefix, C_i) or None - Prefix tokens (including CLS) from the encoder.

        Returns:
            (B, fusion_channels, H_out, W_out) - The final fused dense prediction map (typically 1/2 resolution of the original image).
        """
        processed = []
        
        # --- Stage 1: Readout and Reassemble ---
        for i, (feat, prefix) in enumerate(zip(features, prefix_tokens)):
            
            # CLS token integration
            if self.readout == 'cat' and prefix is not None:
                # (B, C_i, H_i, W_i) -> (B, C_i, H_i, W_i)
                feat = self.readout_blocks[i](feat, prefix)
            elif self.readout == 'add' and prefix is not None:
                # (B, num_prefix, C_i) -> (B, C_i, 1, 1)
                cls = prefix.mean(dim=1).view(prefix.shape[0], prefix.shape[2], 1, 1)
                # (B, C_i, H_i, W_i) + (B, C_i, 1, 1) -> (B, C_i, H_i, W_i)
                feat = feat + cls
            
            # Reassemble to uniform feature dimensions: 
            # (B, C_i, H_i, W_i) -> (B, fusion_channels, H_target, W_target)
            feat = self.reassemble_blocks[i](feat)
            processed.append(feat)

        # --- Stage 2: Progressive Fusion (Coarse to Fine) ---
        fused: torch.Tensor | None = None
        
        # Iterate backwards from the deepest (coarsest resolution) layer up to the shallowest (finest resolution) layer
        for fusion_block, feat in zip(self.fusion_blocks, reversed(processed)):
            # (B, fusion_channels, H, W) -> (B, fusion_channels, 2H, 2W)
            fused = fusion_block(feat, fused)

        assert fused is not None, "Fusion blocks should produce a fused output"
        return fused

    @model_context(requires=['pyramid', 'prefix_tokens'])
    def forward_feature_map(self, ctx: dict) -> dict:
        """Fuse multi-scale ViT features into a single dense map.

        Reads ctx['pyramid'] and ctx['prefix_tokens'].
        Writes 'feature_map' as (B, fusion_channels, H, W).

        Args:
            ctx: Context dict with 'pyramid' and 'prefix_tokens'.

        Returns:
            {'feature_map': tensor}.
        """
        return {'feature_map': self.forward(ctx['pyramid'], ctx['prefix_tokens'])}

