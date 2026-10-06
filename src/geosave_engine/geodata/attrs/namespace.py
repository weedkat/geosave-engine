"""One flat attrs mapping, parsed into the models its scope carries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self, overload

from .model import AttrsModel, FlatAttrs, all_equal
from .models import MODELS, Scope, model_scope, resolve_model, scope_keys


@dataclass(frozen=True)
class AttrsNamespace:
    """One xarray attrs mapping, split into typed models and foreign keys.

    Args:
        models: Model name mapped to the model parsed from the mapping.
        foreign: Keys no present model writes, kept as they came in.

    Raises:
        KeyError: A model name names no GeoSave model.
        TypeError: A model does not match its name.
        ValueError: The models belong to different scopes, or a foreign key is
            written by one of the models.

    Examples:
        >>> namespace = AttrsNamespace.from_attrs(
        ...     {"units": "1", "mission": "S2"}, "variable"
        ... )
        >>> namespace.get(CFVariable).units, namespace.foreign
        ('1', {'mission': 'S2'})
    """

    models: Mapping[str, AttrsModel] = field(default_factory=dict[str, AttrsModel])
    foreign: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        """Refuse a model filed under the wrong name, models of two scopes, or a key claimed twice."""
        for model_name, model in self.models.items():
            model_type = resolve_model(model_name)
            if not isinstance(model, model_type):
                raise TypeError(
                    f"attrs model {model_name!r} must be {model_type.__name__}, "
                    f"got {type(model).__name__}"
                )
        scopes = {model_scope(type(model)) for model in self.models.values()}
        if len(scopes) > 1:
            raise ValueError(
                f"models of different scopes cannot share a namespace: {sorted(scopes)}"
            )
        model_keys = {
            key for model in self.models.values() for key in model.attr_keys()
        }
        if collisions := sorted(self.foreign.keys() & model_keys):
            raise ValueError(f"foreign attrs collide with model keys: {collisions}")

    @property
    def scope(self) -> Scope | None:
        """Return the scope its models belong to, None when it holds none."""
        scopes: set[Scope] = {
            model_scope(type(model)) for model in self.models.values()
        }
        return scopes.pop() if scopes else None

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any], scope: Scope) -> Self:
        """Parse one flat xarray attrs mapping against the models of one scope.

        A model appears where the mapping carries at least one of its keys.
        xarray types attrs keys as merely `Hashable`, but every key actually
        written is a string, so this is where that gets settled once.

        Args:
            attrs: Flat attrs mapping, however its keys are typed.
            scope: Where the mapping lives.

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
        for model_type in MODELS[scope]:
            model = model_type.from_attrs(flat_attrs)
            if model is not None:
                models[model_type.NAME] = model
        keys = scope_keys(scope)
        foreign = {key: value for key, value in flat_attrs.items() if key not in keys}
        return cls(models, foreign)

    @classmethod
    def merge(cls, namespaces: Sequence[AttrsNamespace]) -> tuple[Self, set[str]]:
        """Merge several attrs mappings into the one their combination carries.

        Each model decides for itself what survives, through its own `merge`.
        A foreign key survives only where every mapping carries the same value.

        Args:
            namespaces: The mappings being merged, in call order, at least one.

        Returns:
            (merged namespace, every attr key dropped from it — model and
            foreign alike)

        Raises:
            ValueError: `namespaces` is empty, the merged models belong to
                different scopes, or a model refuses what the mappings disagree
                on.
        """
        if not namespaces:
            raise ValueError("merging attrs needs at least one namespace")
        dropped: set[str] = set()
        merged_models: dict[str, AttrsModel] = {}
        for model_name in sorted(set().union(*(n.models for n in namespaces))):
            models = [namespace.models.get(model_name) for namespace in namespaces]
            merged_models[model_name], dropped_keys = resolve_model(model_name).merge(
                models
            )
            dropped |= dropped_keys

        foreign: dict[str, object] = {}
        for key in sorted(set().union(*(n.foreign for n in namespaces))):
            values = [n.foreign[key] for n in namespaces if key in n.foreign]
            if len(values) == len(namespaces) and all_equal(values):
                foreign[key] = values[0]
            else:
                dropped.add(key)
        return cls(merged_models, foreign), dropped

    def to_attrs(self) -> FlatAttrs:
        """Flatten this namespace back into one xarray attrs mapping.

        Returns:
            {
                "<attr key>": its value,
            }
            Foreign keys first, then every key the models write.
        """
        model_attrs: FlatAttrs = {}
        for model in self.models.values():
            model_attrs.update(model.to_attrs())
        return {**self.foreign, **model_attrs}

    @overload
    def get[M: AttrsModel](self, model: type[M]) -> M | None: ...

    @overload
    def get(self, model: str) -> AttrsModel | None: ...

    def get[M: AttrsModel](self, model: type[M] | str) -> M | AttrsModel | None:
        """Read one model by its class or `NAME`.

        Args:
            model: Model class or its `NAME`.

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
