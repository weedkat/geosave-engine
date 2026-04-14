from __future__ import annotations
from typing import Any

class BaseGeosaveModel:
    """
    Base class for all models in the GeoSave Engine.
    """
    task: dict[str, list[str]] # value is a list of methods
    model: Any

    def __init__(self, *args: Any, **kwargs: Any):
        self.model_instance = self.model(*args, **kwargs)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_instance(*args, **kwargs)
    



