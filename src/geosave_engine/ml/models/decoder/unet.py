from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from geosave_engine.ml.models.contract import ModelContext, model_context


class _ChannelProject(nn.Module):
    """
    Lightweight adapter for CNN/Swin backbones. 
    Only aligns channel dimensions; leaves spatial dimensions exactly as they are.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            (B, out_channels, H, W) - Same spatial resolution, new channel depth.
        """
        return self.proj(x)


class _Reassemble(nn.Module):
    """
    The "ViT Adapter" block. Projects a feature map to the correct U-Net channel width 
    and forcefully resamples it to fake a multi-scale pyramid stride.

    Args:
        in_channels: Feature channels from the encoder.
        out_channels: Target channels for the U-Net skip connection.
        src_stride: The spatial stride the feature map currently has (e.g., 16 for standard ViT).
        tgt_stride: The spatial stride the U-Net needs at this pyramid level (e.g., 4, 8, 16, or 32).
    """
    def __init__(self, in_channels: int, out_channels: int, src_stride: int, tgt_stride: int):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if tgt_stride < src_stride:
            scale = src_stride // tgt_stride
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=scale, stride=scale
            )
        elif tgt_stride > src_stride:
            scale = tgt_stride // src_stride
            self.resample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=scale, padding=1
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H_src, W_src)

        Returns:
            (B, out_channels, H_tgt, W_tgt)
        """
        # (B, in_channels, H_src, W_src) -> (B, out_channels, H_tgt, W_tgt)
        return self.resample(self.proj_in(x))


class _DoubleConv(nn.Module):
    """
    Standard U-Net convolutional block: two sequential 3x3 Convolutions, each followed by Norm and ReLU.

    Args:
        in_channels: Channel dimension of the input.
        out_channels: Desired channel dimension of the output.
        use_norm: Whether to apply BatchNorm2d.
    """
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_norm)
        self.norm1 = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_norm)
        self.norm2 = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        
        Returns:
            (B, out_channels, H, W)
        """
        # (B, in_channels, H, W) -> (B, out_channels, H, W)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class _UpBlock(nn.Module):
    """
    One expanding step of the U-Net decoder: Upsample -> Concat Skip Connection -> Double Conv.

    Args:
        in_channels: Channel dimension coming from the previous (deeper) decoder block.
        skip_channels: Channel dimension coming horizontally from the encoder (Reassemble block).
        out_channels: Channel dimension to output for the next block.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        self.conv = _DoubleConv(in_channels + skip_channels, out_channels, use_norm=use_norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) - Features from the deeper decoder stage.
            skip: (B, skip_channels, 2H, 2W) or None - Finer features from the Reassemble block.
        
        Returns:
            (B, out_channels, 2H, 2W) - Upsampled and fused feature map.
        """
        # (B, in_channels, H, W) -> (B, in_channels, 2H, 2W)
        target_hw = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        
        if skip is not None:
            # (B, in_channels, 2H, 2W) + (B, skip_channels, 2H, 2W) -> (B, combined, 2H, 2W)
            x = torch.cat([x, skip], dim=1)
            
        # (B, combined, 2H, 2W) -> (B, out_channels, 2H, 2W)
        return self.conv(x)


class UnetDecoder(nn.Module):
    """
    U-Net Decoder with a toggleable `vit_adapter` parameter.
    """
    def __init__(
        self,
        encoder_out_channels: list[int],
        encoder_output_strides: list[int],
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
        use_norm: bool = True,
        vit_adapter: bool = True,
    ):
        super().__init__()
        n = len(encoder_out_channels)
        
        # Calculate target channels to align with U-Net skips
        reassemble_channels = [
            decoder_channels[-(i + 1)] if i < n - 1 else decoder_channels[0] for i in range(n)
        ]

        # --- 1. Encoder Integration (ViT vs CNN routing) ---
        if vit_adapter:
            # Fake Pyramid Generation for ViTs
            target_strides = [2 ** (i + 2) for i in range(n)]
            self.encoder_adapter = nn.ModuleList([
                _Reassemble(
                    in_channels=encoder_out_channels[i],
                    out_channels=reassemble_channels[i],
                    src_stride=encoder_output_strides[i],
                    tgt_stride=target_strides[i],
                ) for i in range(n)
            ])
        else:
            # Lightweight Channel Projection for hierarchical CNNs/Swin
            self.encoder_adapter = nn.ModuleList([
                _ChannelProject(
                    in_channels=encoder_out_channels[i],
                    out_channels=reassemble_channels[i],
                ) for i in range(n)
            ])

        # --- 2. Build U-Net UpBlocks ---
        self.up_blocks = nn.ModuleList()
        in_ch = reassemble_channels[-1] 
        
        for i in range(n - 1):
            skip_ch = reassemble_channels[n - 2 - i]
            out_ch = decoder_channels[i]
            self.up_blocks.append(_UpBlock(in_ch, skip_ch, out_ch, use_norm=use_norm))
            in_ch = out_ch

        # --- 3. Final Upsample ---
        self.final_up = _UpBlock(
            in_channels=in_ch, skip_channels=0, out_channels=decoder_channels[-1], use_norm=use_norm
        )

        self.out_channels = decoder_channels[-1]

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Step 1: Route through the correct adapter (Full Reassemble or just Channel Proj)
        # (B, C_in, H_src, W_src) -> (B, reassemble_C, H_tgt, W_tgt)
        pyramid = [block(f) for block, f in zip(self.encoder_adapter, features)]
        
        x = pyramid[-1]

        # Step 2: Progressive Upsampling
        for i, block in enumerate(self.up_blocks):
            skip = pyramid[len(pyramid) - 2 - i]
            x = block(x, skip)

        # Step 3: Final output resolution bump
        x = self.final_up(x, skip=None)

        return x

    @model_context(inputs=(['pyramid'], ['feature_map']))
    def forward_feature_map(self, ctx: ModelContext) -> ModelContext:
        """Fuse multi-scale pyramid into a single dense feature map.

        Reads ctx.inputs['pyramid'] (list of per-level tensors).
        Writes ctx.inputs['feature_map'] as (B, out_channels, H, W).

        Args:
            ctx: ModelContext with 'pyramid' in inputs.

        Returns:
            ModelContext with ctx.inputs = {'feature_map': tensor}.
        """
        fused = self.forward(ctx.inputs['pyramid'])
        return ModelContext(
            inputs={'feature_map': fused},
            sample_meta=ctx.sample_meta,
            metadata=ctx.metadata,
        )

