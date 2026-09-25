"""One flat attrs mapping, parsed into the models it carries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self, overload

from .model import (
    REGISTERED_ATTR_KEYS,
    REGISTERED_MODELS,
    AttrsModel,
    attrs_equal,
    parse_field_value,
    resolve_model,
)


@dataclass(frozen=True)
class AttrsNamespace:
    """One xarray attrs mapping, split into typed models and foreign keys.

    Args:
        models: Registered model name mapped to the model parsed from the
            mapping.
        foreign: Keys no registered model writes, kept as they came in.

    Raises:
        KeyError: A model name is not registered.
        TypeError: A model does not match its registered name.
        ValueError: A foreign key is owned by a registered model.

    Examples:
        >>> namespace = AttrsNamespace.from_attrs({"units": "1", "mission": "S2"})
        >>> namespace.get(CFVariable).units, namespace.foreign
        ('1', {'mission': 'S2'})
    """

    models: Mapping[str, AttrsModel] = field(default_factory=dict[str, AttrsModel])
    foreign: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        """Refuse a model filed under the wrong name, or a key claimed twice.

        Raises:
            KeyError: A model name is not registered.
            TypeError: A model does not match its registered name.
            ValueError: A foreign key is owned by a registered model.
        """
        for model_name, model in self.models.items():
            expected = resolve_model(model_name)
            if not isinstance(model, expected):
                raise TypeError(
                    f"attrs model {model_name!r} must be {expected.__name__}, "
                    f"got {type(model).__name__}"
                )
        collisions = sorted(self.foreign.keys() & REGISTERED_ATTR_KEYS)
        if collisions:
            raise ValueError(
                f"foreign attrs collide with registered keys: {collisions}"
            )

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any]) -> Self:
        """Parse one flat xarray attrs mapping.

        A model appears where the mapping carries at least one of its keys.
        xarray types attrs keys as merely `Hashable`, but every key actually
        written is a string, so this is where that gets settled once.

        Args:
            attrs: Flat attrs mapping, however its keys are typed.

        Returns:
            Namespace holding the models the mapping carries and every key
            none of them writes.

        Raises:
            ValueError: A field's spellings are set to different values.
            ValidationError: A value does not satisfy the field that owns its
                key.
        """
        flat_attrs = {str(key): value for key, value in attrs.items()}

        models: dict[str, AttrsModel] = {}
        for model_name, model_type in REGISTERED_MODELS.items():
            kwargs: dict[str, Any] = {}
            for field_name, attr_keys in model_type.field_keys.items():
                spellings = [key for key in attr_keys if key in flat_attrs]
                if not spellings:
                    continue
                value = flat_attrs[spellings[0]]
                if len(spellings) > 1:
                    # One store may hold a spelling as text and another as a number.
                    read = [
                        parse_field_value(model_type, field_name, flat_attrs[key])
                        for key in spellings
                    ]
                    for spelling, other in zip(spellings[1:], read[1:], strict=True):
                        if not attrs_equal(other, read[0]):
                            raise ValueError(
                                f"{spellings[0]!r} is {read[0]!r} but "
                                f"{spelling!r} is {other!r}; they spell one "
                                f"{model_type.__name__}.{field_name}, so set "
                                f"one of them"
                            )
                kwargs[field_name] = value
            if kwargs:
                models[model_name] = model_type(**kwargs)

        foreign: dict[str, Any] = {}
        for attr_key, value in flat_attrs.items():
            if attr_key not in REGISTERED_ATTR_KEYS:
                foreign[attr_key] = value

        return cls(models=models, foreign=foreign)

    @classmethod
    def merge(cls, namespaces: Sequence[AttrsNamespace]) -> tuple[Self, set[str]]:
        """Merge several attrs mappings into the one their combination carries.

        Each model decides for itself what survives, through its own `merge`.
        A foreign key survives only where every mapping carries the same value.

        Args:
            namespaces: The mappings being merged, in call order, at least one.

        Returns:
            (merged namespace, every attr key dropped from it — registered
            and foreign alike)

        Raises:
            ValueError: `namespaces` is empty.
        """
        if not namespaces:
            raise ValueError("merging attrs needs at least one namespace")

        # Model presence is a view of flat keys, not an independent claim of
        # absence when another model in the same namespace carries that key.
        namespaces = [cls.from_attrs(namespace.to_attrs()) for namespace in namespaces]

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
            elif all(
                attrs_equal(value, values[0]) for value in values[1:]
            ):
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
            ValueError: Models state different values for the same attr key.
        """
        values: dict[str, Any] = {}
        owners: dict[str, str] = {}
        for model in self.models.values():
            for attr_key, value in model.to_attrs().items():
                if attr_key in values and not attrs_equal(values[attr_key], value):
                    raise ValueError(
                        f"attr {attr_key!r} disagrees between models "
                        f"{owners[attr_key]!r} and {model.NAME!r}; "
                        "one namespace must state one value for each key"
                    )
                values[attr_key] = value
                owners[attr_key] = model.NAME
        return {
            **self.foreign,
            **{key: value for key, value in values.items() if value is not None},
        }

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
            >>> namespace.get(Packing)
            Packing(scale_factor=0.0001, add_offset=None)
        """
        if isinstance(model, str):
            return self.models.get(model)
        instance = self.models.get(model.NAME)
        return instance if isinstance(instance, model) else None
