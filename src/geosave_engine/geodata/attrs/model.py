"""Typed metadata models carried in flat xarray attrs mappings."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, KeysView, Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, Self, get_args

from pydantic import BaseModel, ConfigDict

_REGISTRY: dict[str, type[AttrsModel]] = {}

# Attr key (xarray object) mapped to the field that fixed its type, which later ones must match.
_ATTR_KEY_DEFINITIONS: dict[str, tuple[type[AttrsModel], str]] = {}

# Live views onto the registry, filled as models are registered.
REGISTERED_MODELS: Mapping[str, type[AttrsModel]] = MappingProxyType(_REGISTRY)
REGISTERED_ATTR_KEYS: KeysView[str] = _ATTR_KEY_DEFINITIONS.keys()

# rebase takes these as keywords, so no model may answer to them.
RESERVED_NAMES = frozenset({"target", "inplace"})


class AttrsModel(BaseModel):
    """One named group of flat attr keys, read and written as typed fields.

    Each field reads and writes one key in an xarray attrs mapping: the
    field's `alias` where it has one, else the field's own name.

    Attributes:
        NAME: Stable name, unique in the registry, used as `rebase`'s keyword.
        attr_keys: Each field name mapped to the attr key it reads and writes,
            filled in at registration.

    Args:
        **fields: Values for the concrete model's own fields.

    Raises:
        ValidationError: A value does not satisfy its field.

    Examples:
        An aliased field writes the alias, because `_FillValue` is the CF
        spelling; an unaliased one writes its own name:

        >>> Nodata(fill_value=0).to_attrs()
        {'_FillValue': 0, 'nodata': 0}
        >>> Packing(scale_factor=1e-4).to_attrs()
        {'scale_factor': 0.0001}
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    NAME: ClassVar[str]
    attr_keys: ClassVar[Mapping[str, str]]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register a concrete model after Pydantic builds its fields.

        A model needs a unique, unreserved name, at least one field, its own
        `merge`, and field types matching any other model that already
        writes one of its attr keys.

        Args:
            **kwargs: Arguments forwarded to Pydantic's subclass hook.

        Raises:
            ValueError: The name is empty, reserved, or registered; the model
                has no fields or no `merge`; a field splits a validation and
                serialization alias; or a shared attr key is typed differently
                from where it was first defined.
        """
        super().__pydantic_init_subclass__(**kwargs)

        name = cls.__dict__.get("NAME")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(
                f"{cls.__name__} must declare a non-empty NAME without "
                "surrounding whitespace"
            )
        if name in RESERVED_NAMES:
            raise ValueError(
                f"{cls.__name__} may not be named {name!r}; rebase takes that "
                f"as a keyword, so a model answering to it is unreachable"
            )
        if name in _REGISTRY:
            raise ValueError(f"attrs model name {name!r} is already registered")
        if not cls.model_fields:
            raise ValueError(f"{cls.__name__} must carry at least one attrs field")

        # merge is abstract, so a model missing it fails at first use; say so at import.
        if cls.__abstractmethods__:
            raise ValueError(
                f"{cls.__name__} must implement "
                f"{sorted(cls.__abstractmethods__)}; every model writes its own "
                f"merge policy"
            )

        # An alias is the attr key to write; without one the field name is it.
        field_attr_keys = {
            field_name: field.alias or field_name
            for field_name, field in cls.model_fields.items()
        }
        for field_name, field in cls.model_fields.items():
            # Pydantic mirrors a plain alias both ways; a split alias would read one key and write another.
            if not (field.alias == field.validation_alias == field.serialization_alias):
                raise ValueError(
                    f"{cls.__name__}.{field_name} must read and write one flat "
                    "attr key; declare it with alias= instead of a validation "
                    "or serialization alias"
                )

            attr_key = field_attr_keys[field_name]
            definition = _ATTR_KEY_DEFINITIONS.get(attr_key)
            if definition is None:
                continue

            # A shared key must mean one value, however it is read.
            first_model, first_field = definition
            defined = first_model.model_fields[first_field]
            if (
                field.annotation != defined.annotation
                or field.metadata != defined.metadata
            ):
                raise ValueError(
                    f"attr {attr_key!r} is typed one way by "
                    f"{first_model.__name__}.{first_field} and another by "
                    f"{cls.__name__}.{field_name}"
                )
            if _has_field_converter(cls, field_name) or _has_field_converter(
                first_model, first_field
            ):
                raise ValueError(
                    f"attr {attr_key!r} is written by "
                    f"{first_model.__name__}.{first_field} and "
                    f"{cls.__name__}.{field_name}, one of them through a "
                    "model-specific field validator or serializer; put shared "
                    "conversion in one reusable Annotated type"
                )

        # Publish only a model that passed every registration constraint.
        _REGISTRY[name] = cls
        cls.attr_keys = MappingProxyType(field_attr_keys)
        for field_name, attr_key in field_attr_keys.items():
            _ATTR_KEY_DEFINITIONS.setdefault(attr_key, (cls, field_name))

    def to_attrs(self) -> dict[str, Any]:
        """Read this model back as a flat attrs mapping.

        Values come out JSON-native (datetimes as ISO 8601 strings,
        timedeltas as ISO 8601 durations, dict keys as strings) so every attr
        stays writable to zarr and netCDF.

        Returns:
            {
                "<attr key>": its value, None where the field marks the attr
                    absent,
            }
            Only fields that were explicitly set appear.

        Examples:
            >>> CFVariable(units="1").to_attrs()
            {'units': '1'}
        """
        return self.model_dump(
            mode="json",
            by_alias=True,
            exclude_unset=True,
            exclude_computed_fields=True,
        )

    @classmethod
    def _merge_fields(
        cls, models: Sequence[AttrsModel | None], *, must_agree: Iterable[str]
    ) -> tuple[Self, set[str]]:
        """Keep the fields every object of a join set to the same value.

        A field they set differently raises when `must_agree` names it, and
        otherwise drops to None.

        Args:
            models: This model from each joined object, at least one, None
                where an object carried none.
            must_agree: Field names whose disagreement is an error rather than
                a drop.

        Returns:
            (model the joined result carries, attr keys the dropped fields
            write). A dropped field reads as None.

        Raises:
            TypeError: An object carries a different model.
            ValueError: `models` is empty, or the objects set a field named by
                `must_agree` differently.
        """
        if not models:
            raise ValueError(f"merging {cls.NAME} needs at least one object")

        present: list[Self] = []
        wrong_types: set[str] = set()
        for model in models:
            if model is None:
                continue
            if isinstance(model, cls):
                present.append(model)
            else:
                wrong_types.add(type(model).__name__)
        if wrong_types:
            raise TypeError(
                f"merging {cls.NAME} needs {cls.__name__} instances, got "
                f"{sorted(wrong_types)}"
            )

        # An object carrying no model agrees with nothing, so every field drops.
        agreed: dict[str, Any] = {}
        if len(present) == len(models):
            first, *rest = present
            for name in cls.model_fields:
                value = getattr(first, name)
                if all(values_agree(getattr(other, name), value) for other in rest):
                    agreed[name] = value

        refused = [name for name in must_agree if name not in agreed]
        if refused:
            disagreement = []
            for name in refused:
                per_object = [
                    None if model is None else getattr(model, name) for model in models
                ]
                disagreement.append(f"{name}={per_object}")
            raise ValueError(
                f"objects disagree on {cls.NAME}: {'; '.join(disagreement)}; "
                f"rebase one model before joining"
            )

        fields_set: set[str] = set()
        for model in present:
            fields_set.update(model.model_fields_set)

        dropped_fields = sorted(fields_set - agreed.keys())

        combined: dict[str, Any] = {}
        # A dropped field is set to None, clearing the key a model had written.
        for name in dropped_fields:
            if type(None) in get_args(cls.model_fields[name].annotation):
                combined[name] = None
        # A field no object set stays unset, so the result clears nothing.
        for name, value in agreed.items():
            if name in fields_set:
                combined[name] = value
        return cls(**combined), {cls.attr_keys[name] for name in dropped_fields}

    @classmethod
    @abstractmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge this model across the objects of a joining transform.

        Abstract: every model writes its own policy — which fields refuse a
        disagreement, which drop, which accumulate. `_merge_fields` covers
        the refuse-or-drop case.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            (merged model, attr keys it could not keep)

        Raises:
            TypeError: An object carries a different model.
            ValueError: `models` is empty, or the objects disagree on a field
                this model refuses.

        Examples:
            >>> Packing.merge([Packing(scale_factor=0.0001), Packing()])
            Traceback (most recent call last):
            ValueError: objects disagree on packing: scale_factor=[0.0001, None]
        """


def resolve_model(model: type[AttrsModel] | str) -> type[AttrsModel]:
    """Resolve a class or stable name to its registered concrete model.

    Args:
        model: Registered model class or stable model name.

    Returns:
        Registered concrete model class.

    Raises:
        KeyError: No model uses the supplied stable name.
        TypeError: The class is not a registered concrete model.

    Examples:
        >>> resolve_model("acdd")
        <class '...ACDD'>
    """
    if isinstance(model, str):
        try:
            return _REGISTRY[model]
        except KeyError:
            raise KeyError(f"no attrs model is registered as {model!r}") from None
    name = getattr(model, "NAME", None)
    if name is None or _REGISTRY.get(name) is not model:
        raise TypeError(f"{model!r} is not a registered concrete AttrsModel")
    return model


def values_agree(a: object, b: object) -> bool:
    """Whether two attr values are the same, treating two NaNs as equal.

    Plain `==` reads two NaNs as disagreeing, since `nan != nan`.

    Args:
        a: First value.
        b: Second value.

    Returns:
        True when `a` and `b` are the same value.

    Examples:
        >>> values_agree(float("nan"), float("nan"))
        True
    """
    # NaN is the only value unequal to itself, so that doubles as the NaN test.
    return bool(a == b) or (a != a and b != b)


def _has_field_converter(model: type[AttrsModel], field: str) -> bool:
    """Whether a model carries its own validator or serializer for one field.

    Args:
        model: Model to inspect.
        field: Field name to look for.

    Returns:
        True where a field validator or serializer covers `field`, or covers
        every field.
    """
    validators = model.__pydantic_decorators__.field_validators.values()
    serializers = model.__pydantic_decorators__.field_serializers.values()
    return any(
        field in decorator.info.fields or "*" in decorator.info.fields
        for decorator in validators
    ) or any(
        field in decorator.info.fields or "*" in decorator.info.fields
        for decorator in serializers
    )
