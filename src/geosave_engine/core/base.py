from __future__ import annotations

from typing import Any


class BaseModel:
    """Concrete helper base for discoverable model wrappers."""

    doc_links: list[str] | str | None = None
    tasks: dict[str, list[str]] = {}
    model: Any = None

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> Any:
        model_callable = getattr(cls, "model", None)
        if not callable(model_callable):
            raise ValueError(f"{cls.__name__}.model must be defined and callable")
        return model_callable(*args, **kwargs)


class BaseLoss:
    """Concrete helper base for discoverable loss wrappers."""

    doc_links: list[str] | str | None = None
    loss: Any = None

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> Any:
        loss_callable = getattr(cls, "loss", None)
        if not callable(loss_callable):
            raise ValueError(f"{cls.__name__}.loss must be defined and callable")
        return loss_callable(*args, **kwargs)


class BaseOptimizer:
    """Concrete helper base for discoverable optimizer wrappers."""

    doc_links: list[str] | str | None = None
    optimizer: Any = None

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> Any:
        optimizer_callable = getattr(cls, "optimizer", None)
        if not callable(optimizer_callable):
            raise ValueError(f"{cls.__name__}.optimizer must be defined and callable")

        method_config = kwargs.pop("method", {"mode": "default"})

        if not isinstance(method_config, dict):
            raise TypeError(
                f"{cls.__name__}.build expects 'method' to be a mapping, "
                f"got '{type(method_config).__name__}'"
            )

        mode_name_raw = method_config.get("mode")
        if not isinstance(mode_name_raw, str) or not mode_name_raw:
            raise ValueError(
                f"{cls.__name__}.build requires method.mode as a non-empty string"
            )

        mode_name = mode_name_raw
        mode_kwargs = {
            key: value for key, value in method_config.items() if key != "mode"
        }

        overlap = set(kwargs).intersection(mode_kwargs)
        if overlap:
            keys = ", ".join(sorted(overlap))
            raise ValueError(
                f"{cls.__name__}.build received duplicated keys in global and method config: {keys}"
            )

        mode_callable = getattr(cls, mode_name, None)
        if not callable(mode_callable):
            raise ValueError(f"{cls.__name__} does not support optimizer mode '{mode_name}'")
        return mode_callable(*args, **kwargs, **mode_kwargs)
