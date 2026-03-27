from collections import defaultdict


class Registry:
    def __init__(self, registry:dict=None):
        self.registry = registry or defaultdict()

    def register(self, name=None):
        def decorator(func):
            short_name = name or func.__name__
            self.registry[short_name] = func
            return func
        return decorator

    def build(self, name, *args, **kwargs):
        if name not in self.registry:
            raise ValueError(f"'{name}' not found in registry. Available options: {list(self.registry.keys())}")
        func = self.registry[name]
        return func(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other, Registry):
            self.registry.update(other.registry)
        return self
