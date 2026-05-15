def uppercase_keys(d: dict) -> dict:
    """Return a copy of d with all keys uppercased."""
    return {k.upper(): v for k, v in d.items()}


def builder(name: str, config: dict, registry: dict):
    """Direct callable lookup. Use for models and losses."""
    reg = uppercase_keys(registry)
    key = name.upper()
    if key not in reg:
        raise ValueError(f"Unknown '{name}'. Available: {list(registry.keys())}")

    return reg[key](**config)


def method_builder(name: str, config: dict, registry: dict):
    """Dot-notation method dispatch ('key.method'). Use for optimizers and schedulers."""
    if "." not in name:
        raise ValueError(f"Expected 'key.method' format, got '{name}'.")

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
