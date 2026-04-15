from __future__ import annotations

from geosave_engine.losses.cross_entropy_loss import CrossEntropyLoss
from geosave_engine.optimizers.adamw import AdamW

LOSS_FACTORY = [CrossEntropyLoss]
OPTIM_FACTORY = [AdamW]


# ==== GEOSAVE AUTO-GENERATED FACTORY BLOCK: START ====

_RAW_OPTIM_FACTORY = globals().get('OPTIM_FACTORY', []) or []
_RAW_LOSS_FACTORY = globals().get('LOSS_FACTORY', []) or []

OPTIMIZER_FACTORY = {cls.__name__: cls for cls in _RAW_OPTIM_FACTORY}
LOSS_FACTORY = {cls.__name__: cls for cls in _RAW_LOSS_FACTORY}


def _available_methods(factory_cls):
    return sorted([name for name, value in factory_cls.__dict__.items() if isinstance(value, classmethod) and not name.startswith('_')])


def _resolve_factory_callable(registry, kind, name, method=None, default_method=None):
    factory_cls = registry.get(name)
    if factory_cls is None:
        raise ValueError(f"Unknown {kind} '{name}'. Available: {', '.join(sorted(registry.keys()))}")
    if method is not None:
        candidate = getattr(factory_cls, method, None)
        if callable(candidate):
            return candidate
        raise ValueError(f"{kind.capitalize()} '{name}' does not support method '{method}'. Available methods: {_available_methods(factory_cls)}")
    if default_method is not None:
        candidate = getattr(factory_cls, default_method, None)
        if callable(candidate):
            return candidate
    build_candidate = getattr(factory_cls, 'build', None)
    if callable(build_candidate):
        return build_candidate
    raise ValueError(f"{kind.capitalize()} '{name}' has no build method. Available methods: {_available_methods(factory_cls)}")


def build_loss(name, *args, method='full', **kwargs):
    factory_callable = _resolve_factory_callable(LOSS_FACTORY, 'loss', name, method=method, default_method='full')
    return factory_callable(*args, **kwargs)


def build_optimizer(name, *args, method='full', **kwargs):
    factory_callable = _resolve_factory_callable(OPTIMIZER_FACTORY, 'optimizer', name, method=method, default_method='full')
    return factory_callable(*args, **kwargs)

from geosave_engine.models.dpt.build import DensePredictionTransformer
from geosave_engine.models.smp.build import Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, FPN, PSPNet, PAN, Linknet, MAnet

MODEL_FACTORY = {
    'DensePredictionTransformer': DensePredictionTransformer,
    'Unet': Unet,
    'UnetPlusPlus': UnetPlusPlus,
    'DeepLabV3': DeepLabV3,
    'DeepLabV3Plus': DeepLabV3Plus,
    'FPN': FPN,
    'PSPNet': PSPNet,
    'PAN': PAN,
    'Linknet': Linknet,
    'MAnet': MAnet,
}

def build_model(name, *args, method=None, **kwargs):
    factory_callable = _resolve_factory_callable(MODEL_FACTORY, 'model', name, method=method)
    return factory_callable(*args, **kwargs)

# ==== GEOSAVE AUTO-GENERATED FACTORY BLOCK: END ====
