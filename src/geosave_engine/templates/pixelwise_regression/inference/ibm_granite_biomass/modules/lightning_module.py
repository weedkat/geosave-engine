from __future__ import annotations

import torch
from lightning import LightningModule

from geosave_engine.ml.core.transforms import ImageProcessor
from geosave_engine.ml.models.monolith.ibm_granite_biomass import GraniteGeospatialBiomass


class GraniteBiomassInference(LightningModule):
    """Inference-only Lightning module for above-ground biomass estimation.

    Loads pretrained IBM Granite Biomass weights, normalizes input via the
    model's own ``img_mean`` / ``img_std``, and runs sliding-window inference.

    No training, validation, or test steps are implemented.

    Args:
        patch_size: Spatial patch size for sliding-window inference.
        overlap_ratio: Patch overlap fraction in [0, 1).
        pad_size: Reflect-padding added on each side before patching.
    """

    def __init__(
        self,
        patch_size: int = 224,
        overlap_ratio: float = 0.5,
        pad_size: int = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.pad_size = pad_size

    def configure_model(self) -> None:
        if hasattr(self, "model"):
            return
        self.model = GraniteGeospatialBiomass(pretrained=True)
        self.preprocessor = ImageProcessor(model=self.model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize then run sliding-window biomass regression.

        Args:
            image: (B, 6, H, W) HLS S30 DN values (reflectance × 10000).

        Returns:
            (B, 1, H, W) above-ground biomass predictions.
        """
        image = self.preprocessor(image)
        return self.model.forward_sliding(
            image, self.patch_size, self.overlap_ratio, self.pad_size
        )

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """Run inference on one batch, pass through spatial metadata.

        Args:
            batch: Dict with ``'image'`` as (B, 6, H, W) HLS S30 DN values.

        Returns:
            {
                'prediction': (B, 1, H, W) biomass estimates,
                'crs': list[str] | None,
                'transform': list | None,
                'coordinate': list[tuple[float, float]] | None,
            }
        """
        return {
            'prediction': self(batch['image']),
            'crs': batch.get('crs'),
            'transform': batch.get('transform'),
            'coordinate': batch.get('coordinate'),
        }

    def configure_optimizers(self):
        raise NotImplementedError("GraniteBiomassInference is inference-only")
