from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BuildSpec(BaseModel):
    """Select a registered factory or importable class for construction."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1)
    class_path: str | None = None
    init_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        """Require exactly one valid construction selector."""
        if (self.name is None) == (self.class_path is None):
            raise ValueError("Specify exactly one of name or class_path")
        if self.class_path is not None and "." not in self.class_path:
            raise ValueError(
                "class_path must include an importable module and class name"
            )
        return self

    def resolve[T](
        self,
        registry: Mapping[str, Callable[..., T]],
        base: type[T],
    ) -> Callable[..., T]:
        """Resolve this selector to a compatible factory.

        Args:
            registry: Available named factories.
            base: Required base class for imported classes.

        Returns:
            Selected registered factory or imported class.

        Raises:
            KeyError: If a registered name is unknown.
            AttributeError: If the imported module has no requested class.
            TypeError: If an imported class has the wrong base type.
        """
        if self.name is not None:
            factories = {key.casefold(): factory for key, factory in registry.items()}
            try:
                return factories[self.name.casefold()]
            except KeyError:
                raise KeyError(
                    f"Unknown name {self.name!r}; available: {list(registry)}"
                ) from None

        assert self.class_path is not None
        module_name, _, attribute = self.class_path.rpartition(".")
        factory = getattr(import_module(module_name), attribute)
        if not isinstance(factory, type) or not issubclass(factory, base):
            raise TypeError(
                f"{self.class_path!r} must name a {base.__name__} subclass"
            )
        return factory
