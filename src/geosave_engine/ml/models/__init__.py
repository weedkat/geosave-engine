"""Importing this package runs every @register_model decorator, populating
geosave_engine.ml.core.factory.MODEL_REGISTRY. build_model imports this
package lazily before any registry lookup — see factory._resolve_stage_cls.
"""
from geosave_engine.ml.models.decoder.dpt import DPTDecoder
from geosave_engine.ml.models.decoder.unet import UnetDecoder
from geosave_engine.ml.models.encoder.dinov3 import DINOv3
from geosave_engine.ml.models.head.dense import DenseHead

__all__ = ["DINOv3", "UnetDecoder", "DPTDecoder", "DenseHead"]
