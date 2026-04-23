"""ML-side utility helpers.

Torch-dependent modules (``pretrained``, ``torch_params``) are deliberately NOT
re-exported here so that importing ``geosave_engine.utils.ml`` does not pull
torch into CLI startup. Import them from their submodule when needed:

    from geosave_engine.utils.ml.pretrained import download_weights
    from geosave_engine.utils.ml.torch_params import split_encoder_decoder_params
"""
from geosave_engine.utils.ml.resolver import (
    instantiate_from_config,
    instantiate_from_config_build,
    instantiate_optimizers_from_config,
    resolve_class,
)
from geosave_engine.utils.ml.yaml_config import (
    inject_into_file,
    inject_value,
    load_yaml,
    save_yaml,
)

__all__ = [
    "inject_into_file",
    "inject_value",
    "instantiate_from_config",
    "instantiate_from_config_build",
    "instantiate_optimizers_from_config",
    "load_yaml",
    "resolve_class",
    "save_yaml",
]
