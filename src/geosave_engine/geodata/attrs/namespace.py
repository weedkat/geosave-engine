"""One flat attrs mapping, parsed into the models it states."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self, overload

from .model import (
    REGISTERED_ATTR_KEYS,
    REGISTERED_MODELS,
    AttrsModel,
    resolve_model,
    values_agree,
)


@dataclass(frozen=True)
class AttrsNamespace:
    """Typed and foreign fields in one xarray attrs mapping.

    Args:
        models: Registered model names mapped to their stated models.
        foreign: Fields no registered model writes.
    """

    models: Mapping[str, AttrsModel] = field(default_factory=dict[str, AttrsModel])
    foreign: Mapping[str, object] = field(default_factory=dict[str, object])

    @classmethod
    def from_attrs(cls, attrs: Mapping[Hashable, object]) -> Self:
        """Parse one flat xarray attrs mapping.

        Args:
            attrs: Flat xarray attrs mapping.

        Returns:
            Namespace holding stated registered models and foreign fields.

        Raises:
            ValidationError: A stated registered field is invalid.
        """
        flat_attrs = {str(key): value for key, value in attrs.items()}

        models: dict[str, AttrsModel] = {}
        for model_name, model_type in REGISTERED_MODELS.items():
            stated: dict[str, object] = {}
            for attr_key in model_type.attr_keys.values():
                if attr_key in flat_attrs:
                    stated[attr_key] = flat_attrs[attr_key]
            if stated:
                models[model_name] = model_type(**stated)

        foreign: dict[str, object] = {}
        for attr_key, value in flat_attrs.items():
            if attr_key not in REGISTERED_ATTR_KEYS:
                foreign[attr_key] = value

        return cls(models=models, foreign=foreign)

    @classmethod
    def combine(cls, namespaces: Sequence[AttrsNamespace]) -> tuple[Self, set[str]]:
        """Combine the namespaces the joined objects carried in one spot.

        Each model decides for itself what survives, through its own `combine`.
        A foreign field survives only where every object states it identically.

        Args:
            namespaces: That spot's namespace from each object being joined, in
                call order, at least one.

        Returns:
            Combined namespace and every attr key dropped from it, registered
            and foreign alike.

        Raises:
            ValueError: `namespaces` is empty, or a registered model refuses a
                disagreement.
        """
        if not namespaces:
            raise ValueError("combining attrs needs at least one namespace")

        model_names: set[str] = set()
        foreign_keys: set[str] = set()
        for namespace in namespaces:
            model_names.update(namespace.models)
            foreign_keys.update(namespace.foreign)

        dropped: set[str] = set()

        models: dict[str, AttrsModel] = {}
        for model_name in sorted(model_names):
            stated_models = [
                namespace.models.get(model_name) for namespace in namespaces
            ]
            models[model_name], dropped_keys = resolve_model(model_name).combine(
                stated_models
            )
            dropped.update(dropped_keys)

        foreign: dict[str, object] = {}
        for attr_key in sorted(foreign_keys):
            stated = [
                namespace.foreign[attr_key]
                for namespace in namespaces
                if attr_key in namespace.foreign
            ]
            if len(stated) < len(namespaces):
                dropped.add(attr_key)
            elif all(values_agree(value, stated[0]) for value in stated[1:]):
                foreign[attr_key] = stated[0]
            else:
                dropped.add(attr_key)

        return cls(models=models, foreign=foreign), dropped

    def to_attrs(self) -> dict[str, Any]:
        """Serialize this namespace into one flat attrs mapping.

        Returns:
            Foreign and registered fields as xarray attrs.

        Raises:
            KeyError: A model name is not registered.
            TypeError: A model does not match its registered name.
            ValueError: A foreign key is owned by a registered model.
        """
        collisions = sorted(self.foreign.keys() & REGISTERED_ATTR_KEYS)
        if collisions:
            raise ValueError(
                f"foreign attrs collide with registered keys: {collisions}"
            )

        attrs = dict(self.foreign)
        for model_name, model in self.models.items():
            expected = resolve_model(model_name)
            if not isinstance(model, expected):
                raise TypeError(
                    f"header model {model_name!r} must be {expected.__name__}, "
                    f"got {type(model).__name__}"
                )
            # A field stating None says the attr is absent, so it writes nothing.
            for attr_key, value in model.to_attrs().items():
                if value is not None:
                    attrs[attr_key] = value
        return attrs

    @overload
    def get[M: AttrsModel](self, model: type[M]) -> M | None: ...

    @overload
    def get(self, model: str) -> AttrsModel | None: ...

    def get[M: AttrsModel](self, model: type[M] | str) -> M | AttrsModel | None:
        """Read one model by its class or registered name.

        Args:
            model: Registered model class or its stable `NAME`.

        Returns:
            Stated model, or None when this namespace does not carry it.
        """
        if isinstance(model, str):
            return self.models.get(model)
        carried = self.models.get(model.NAME)
        return carried if isinstance(carried, model) else None
