"""One flat attrs mapping, parsed into the models its scope carries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self, overload

from .model import AttrsModel, FlatAttrs, all_equal
from .models import MODELS, Scope, model_scope, resolve_model, scope_keys


@dataclass(frozen=True)
class AttrsNamespace:
    """One xarray attrs mapping, split into typed models and foreign keys.

    Args:
        models: Model class mapped to the instance parsed from the mapping.
        foreign: Keys no present model writes, kept as they came in.

    Raises:
        TypeError: A model class is unregistered, or an instance does not
            match the class it is filed under.
        ValueError: The models belong to different scopes, or a foreign key is
            written by one of the models.

    Examples:
        >>> namespace = AttrsNamespace.from_attrs(
        ...     {"units": "1", "mission": "S2"}, "variable"
        ... )
        >>> namespace.get(CFVariable).units, namespace.foreign
        ('1', {'mission': 'S2'})
    """

    models: Mapping[type[AttrsModel], AttrsModel] = field(
        default_factory=dict[type[AttrsModel], AttrsModel]
    )
    foreign: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        """Check model identity, one scope, and distinct owned and foreign keys."""
        scopes: set[Scope] = set()
        model_keys: set[str] = set()
        for model_type, model in self.models.items():
            if not isinstance(model_type, type):
                raise TypeError("attrs models must be keyed by their classes")
            resolve_model(model_type)
            if not isinstance(model, model_type):
                raise TypeError(
                    f"attrs model must be {model_type.__name__}, "
                    f"got {type(model).__name__}"
                )
            scopes.add(model_scope(type(model)))
            model_keys.update(model_type.attr_keys())
        if len(scopes) > 1:
            raise ValueError(
                f"models of different scopes cannot share a namespace: {sorted(scopes)}"
            )
        if collisions := sorted(self.foreign.keys() & model_keys):
            raise ValueError(f"foreign attrs collide with model keys: {collisions}")

    @property
    def scope(self) -> Scope | None:
        """Return the scope its models belong to, None when it holds none."""
        for model_type in self.models:
            return model_scope(model_type)
        return None

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any], scope: Scope) -> Self:
        """Parse one flat xarray attrs mapping against the models of one scope.

        Each registered model reads the same mapping independently and selects
        its own fields. Keys owned by no model in this scope remain foreign
        attrs; the input mapping is untouched.
        xarray types attrs keys as merely `Hashable`, but every key actually
        written is a string, so this is where that gets settled once.

        Args:
            attrs: Flat attrs mapping, however its keys are typed.
            scope: Where the mapping lives.

        Returns:
            Namespace with `models={ModelClass: instance, ...}` and
            `foreign={unclaimed_key: value, ...}`.

        Raises:
            ValueError: A field's spellings are set to different values.
            ValidationError: A value does not satisfy the field that owns its
                key.
        """
        flat_attrs = {str(key): value for key, value in attrs.items()}
        models: dict[type[AttrsModel], AttrsModel] = {}
        for model_type in MODELS[scope].values():
            model = model_type.from_attrs(flat_attrs)
            if model is not None:
                models[model_type] = model

        owned_keys = scope_keys(scope)
        foreign: FlatAttrs = {}
        for key, value in flat_attrs.items():
            if key not in owned_keys:
                foreign[key] = value
        return cls(models, foreign)

    @classmethod
    def merge(
        cls,
        namespaces: Sequence[AttrsNamespace],
        *,
        conflicts: Literal["raise", "drop"] = "raise",
    ) -> tuple[Self, set[str]]:
        """Merge several attrs mappings into the one their combination carries.

        Each model decides for itself what survives, through its own `merge`.
        A foreign key survives only where every mapping carries the same value.

        Args:
            namespaces: The mappings being merged, in call order, at least one.
            conflicts: Whether differing `MUST_AGREE` fields raise or drop.

        Returns:
            (merged namespace, every attr key dropped from it — model and
            foreign alike)

        Raises:
            ValueError: `namespaces` is empty, the merged models belong to
                different scopes, or a model refuses what the mappings disagree
                on when `conflicts="raise"`.
        """
        if not namespaces:
            raise ValueError("merging attrs needs at least one namespace")
        dropped: set[str] = set()
        model_types: set[type[AttrsModel]] = set()
        foreign_keys: set[str] = set()
        for namespace in namespaces:
            model_types.update(namespace.models)
            foreign_keys.update(namespace.foreign)
        merged_models: dict[type[AttrsModel], AttrsModel] = {}
        for model_type in sorted(model_types, key=lambda model: model.__name__):
            models = [namespace.models.get(model_type) for namespace in namespaces]
            merged_models[model_type], dropped_keys = model_type.merge(
                models, conflicts=conflicts
            )
            dropped |= dropped_keys

        foreign: dict[str, object] = {}
        for key in sorted(foreign_keys):
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
        """Read one model by its class or registered configuration name.

        Args:
            model: Model class or its name in the scope registry.

        Returns:
            The model, or None when this namespace does not carry it.

        Examples:
            >>> namespace.get(Packing)
            Packing(scale_factor=0.0001, add_offset=None)
        """
        if isinstance(model, str):
            try:
                model_type = resolve_model(model)
            except KeyError:
                return None
            return self.models.get(model_type)
        instance = self.models.get(model)
        return instance if isinstance(instance, model) else None
