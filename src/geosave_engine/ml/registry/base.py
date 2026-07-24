from __future__ import annotations


def uppercase_keys(d: dict) -> dict:
    """Return copy of ``d`` with all keys uppercased."""
    return {k.upper(): v for k, v in d.items()}


def builder(name: str, config: dict, registry: dict):
    """Instantiate entry from registry by name.

    Case-insensitive. Use for losses and models.

    Args:
        name: Registry key (e.g. ``"CELoss"``).
        config: Keyword args passed to the constructor.
        registry: Mapping of name → callable.

    Returns:
        Instantiated object from registry.

    Raises:
        ValueError: If ``name`` not found in registry.
    """
    reg = uppercase_keys(registry)
    key = name.upper()
    if key not in reg:
        raise ValueError(f"Unknown '{name}'. Available: {list(registry.keys())}")
    return reg[key](**config)


def method_builder(name: str, config: dict, registry: dict):
    """Instantiate via dot-notation method dispatch (``"key.method"``).

    Use for optimizers where variant is a module-level function.
    If no method given, falls back to ``default``.

    Args:
        name: ``"key"`` or ``"key.method"`` (e.g. ``"AdamW.split"``).
        config: Keyword args passed to the method.
        registry: Mapping of name → module with callable methods.

    Returns:
        Result of calling ``registry[key].method(**config)``.

    Raises:
        ValueError: If key or method not found in registry.
    """
    if "." not in name:
        name = name + ".default"

    reg = uppercase_keys(registry)
    raw_key, method = name.split(".", 1)
    key = raw_key.upper()
    if key not in reg:
        raise ValueError(f"Unknown key '{raw_key}'. Available: {list(registry.keys())}")

    entry = reg[key]
    if not hasattr(entry, method):
        available = [m for m in dir(entry) if not m.startswith("_")]
        raise ValueError(f"Unknown method '{method}' on '{raw_key}'. Available: {available}")

    return getattr(entry, method)(**config)
