"""One flat attrs mapping, parsed into the models it carries."""

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
    """One xarray attrs mapping, split into typed models and foreign keys.

    Args:
        models: Registered model name mapped to the model parsed from the
            mapping.
        foreign: Keys no registered model writes, kept as they came in.

    Examples:
        >>> namespace = AttrsNamespace.from_attrs({"units": "1", "mission": "S2"})
        >>> namespace.get(CFVariable).units, namespace.foreign
        ('1', {'mission': 'S2'})
    """

    models: Mapping[str, AttrsModel] = field(default_factory=dict[str, AttrsModel])
    foreign: Mapping[str, object] = field(default_factory=dict[str, object])

    @classmethod
    def from_attrs(cls, attrs: Mapping[Hashable, Any]) -> Self:
        """Parse one flat xarray attrs mapping.

        A model appears where the mapping carries at least one of its keys.
        xarray types attrs keys as merely `Hashable`, but every key actually
        written is a string, so this is where that gets settled once.

        Args:
            attrs: Flat xarray attrs mapping.

        Returns:
            Namespace holding the models the mapping carries and every key
            none of them writes.

        Raises:
            ValidationError: A value does not satisfy the field that owns its
                key.
        """
        flat_attrs = {str(key): value for key, value in attrs.items()}

        models: dict[str, AttrsModel] = {}
        for model_name, model_type in REGISTERED_MODELS.items():
            kwargs: dict[str, Any] = {}
            for attr_key in model_type.attr_keys.values():
                if attr_key in flat_attrs:
                    kwargs[attr_key] = flat_attrs[attr_key]
            if kwargs:
                models[model_name] = model_type(**kwargs)

        foreign: dict[str, Any] = {}
        for attr_key, value in flat_attrs.items():
            if attr_key not in REGISTERED_ATTR_KEYS:
                foreign[attr_key] = value

        return cls(models=models, foreign=foreign)

    @classmethod
    def merge(cls, namespaces: Sequence[AttrsNamespace]) -> tuple[Self, set[str]]:
        """Merge the namespaces the joined objects carried at one name.

        Each model decides for itself what survives, through its own `merge`.
        A foreign key survives only where every object carries the same value.

        Args:
            namespaces: That name's namespace from each object being joined,
                in call order, at least one.

        Returns:
            (merged namespace, every attr key dropped from it — registered
            and foreign alike)

        Raises:
            ValueError: `namespaces` is empty, or a registered model refuses a
                disagreement.
        """
        if not namespaces:
            raise ValueError("merging attrs needs at least one namespace")

        model_names: set[str] = set()
        foreign_keys: set[str] = set()
        for namespace in namespaces:
            model_names.update(namespace.models)
            foreign_keys.update(namespace.foreign)

        dropped: set[str] = set()

        models: dict[str, AttrsModel] = {}
        for model_name in sorted(model_names):
            per_object = [namespace.models.get(model_name) for namespace in namespaces]
            models[model_name], dropped_keys = resolve_model(model_name).merge(
                per_object
            )
            dropped.update(dropped_keys)

        foreign: dict[str, object] = {}
        for attr_key in sorted(foreign_keys):
            values = [
                namespace.foreign[attr_key]
                for namespace in namespaces
                if attr_key in namespace.foreign
            ]
            if len(values) < len(namespaces):
                dropped.add(attr_key)
            elif all(values_agree(value, values[0]) for value in values[1:]):
                foreign[attr_key] = values[0]
            else:
                dropped.add(attr_key)

        return cls(models=models, foreign=foreign), dropped

    def to_attrs(self) -> dict[str, Any]:
        """Flatten this namespace back into one xarray attrs mapping.

        Returns:
            {
                "<attr key>": its value,
            }
            Foreign keys first, then every key the models write. A field set
            to None writes nothing, marking its key absent.

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
            # A field set to None marks the attr absent, so it writes nothing.
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
            The model, or None when this namespace does not carry it.

        Examples:
            >>> namespace.get(Packing).scale_factor
            0.0001
        """
        if isinstance(model, str):
            return self.models.get(model)
        instance = self.models.get(model.NAME)
        return instance if isinstance(instance, model) else None
