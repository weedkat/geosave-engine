"""Typed metadata models carried in flat xarray attrs mappings."""

from __future__ import annotations

from collections.abc import KeysView, Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, Self, get_args

from pydantic import BaseModel, ConfigDict, TypeAdapter
from xarray.core.duck_array_ops import array_equiv

_MODEL_TYPES: dict[str, type[AttrsModel]] = {}

# Each xarray attr key is owned by the first model field that defines its type.
_FIELD_BY_ATTR_KEY: dict[str, tuple[type[AttrsModel], str]] = {}

# Live views onto the registry, filled as models are registered.
REGISTERED_MODELS: Mapping[str, type[AttrsModel]] = MappingProxyType(_MODEL_TYPES)
REGISTERED_ATTR_KEYS: KeysView[str] = _FIELD_BY_ATTR_KEY.keys()

# rebase takes these as keywords, so no model may answer to them.
_RESERVED_MODEL_NAMES = frozenset({"target", "inplace"})


class AttrsModel(BaseModel):
    """One named group of flat attr keys, read and written as typed fields.

    Each field reads and writes its own name in an xarray `.attrs` mapping
    unless `field_keys` declares other keys.

    Attributes:
        NAME: Stable name, unique in the registry, used as `rebase`'s keyword.
        field_keys: Model field names mapped to key names in a flat xarray
            `.attrs` dictionary. It maps names, not attr values. Each field
            may use more than one key; for example, `Nodata.fill_value` uses
            `("_FillValue", "nodata")`.

    Args:
        **fields: Values for the concrete model's own fields.

    Raises:
        ValidationError: A value does not satisfy its field.

    Examples:
        A field with several keys writes each of them; a plain field writes its
        own name:

        >>> Nodata(fill_value=0).to_attrs()
        {'_FillValue': 0, 'nodata': 0}
        >>> Packing(scale_factor=1e-4).to_attrs()
        {'scale_factor': 0.0001}
    """

    # A reader may type a text tag by how it reads, so a string field takes a number.
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        coerce_numbers_to_str=True,
    )

    NAME: ClassVar[str]
    field_keys: ClassVar[Mapping[str, tuple[str, ...]]]
    _field_parsers: ClassVar[Mapping[str, TypeAdapter[Any]]]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register a concrete model after Pydantic builds its fields.

        A model needs a unique, unreserved name, at least one field,
        and field types matching any other model that already
        writes one of its attr keys.

        Args:
            **kwargs: Arguments forwarded to Pydantic's subclass hook.

        Raises:
            ValueError: The name is empty, reserved, or registered; the model
                has no fields; or a shared attr key is typed differently from
                where it was first defined.
        """
        super().__pydantic_init_subclass__(**kwargs)

        model_name = cls.__dict__.get("NAME")
        if (
            not isinstance(model_name, str)
            or not model_name
            or model_name != model_name.strip()
        ):
            raise ValueError(
                f"{cls.__name__} must declare a non-empty NAME without "
                "surrounding whitespace"
            )
        if model_name in _RESERVED_MODEL_NAMES:
            raise ValueError(
                f"{cls.__name__} may not be named {model_name!r}; rebase takes that "
                f"as a keyword, so a model answering to it is unreachable"
            )
        if model_name in _MODEL_TYPES:
            raise ValueError(
                f"attrs model name {model_name!r} is already registered"
            )
        if not cls.model_fields:
            raise ValueError(f"{cls.__name__} must carry at least one attrs field")

        custom_keys = dict(getattr(cls, "field_keys", {}))
        unknown_fields = custom_keys.keys() - cls.model_fields.keys()
        if unknown_fields:
            raise ValueError(
                f"{cls.__name__}.field_keys names unknown fields "
                f"{sorted(unknown_fields)}"
            )

        field_keys: dict[str, tuple[str, ...]] = {}
        for field_name in cls.model_fields:
            attr_keys = custom_keys.get(field_name, (field_name,))
            if (
                not isinstance(attr_keys, tuple)
                or not attr_keys
                or any(not isinstance(key, str) or not key for key in attr_keys)
            ):
                raise ValueError(
                    f"{cls.__name__}.field_keys[{field_name!r}] must be a "
                    "nonempty tuple of xarray attr key names"
                )
            field_keys[field_name] = attr_keys

        for field_name, field in cls.model_fields.items():
            for attr_key in field_keys[field_name]:
                owner = _FIELD_BY_ATTR_KEY.get(attr_key)
                if owner is None:
                    continue

                # A shared key must mean one value, however it is read.
                owner_model, owner_field_name = owner
                owner_field = owner_model.model_fields[owner_field_name]
                if (
                    field.annotation != owner_field.annotation
                    or field.metadata != owner_field.metadata
                ):
                    raise ValueError(
                        f"attr {attr_key!r} is typed one way by "
                        f"{owner_model.__name__}.{owner_field_name} and another by "
                        f"{cls.__name__}.{field_name}"
                    )
                if _field_has_converter(cls, field_name) or _field_has_converter(
                    owner_model, owner_field_name
                ):
                    raise ValueError(
                        f"attr {attr_key!r} is written by "
                        f"{owner_model.__name__}.{owner_field_name} and "
                        f"{cls.__name__}.{field_name}, one of them through a "
                        "model-specific field validator or serializer; put shared "
                        "conversion in one reusable Annotated type"
                    )

        # Publish only a model that passed every registration constraint.
        _MODEL_TYPES[model_name] = cls
        cls.field_keys = MappingProxyType(field_keys)
        cls._field_parsers = MappingProxyType(
            {
                field_name: TypeAdapter(field.rebuild_annotation())
                for field_name, field in cls.model_fields.items()
            }
        )
        for field_name, attr_keys in field_keys.items():
            for attr_key in attr_keys:
                _FIELD_BY_ATTR_KEY.setdefault(attr_key, (cls, field_name))

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
            Only fields that were explicitly set appear, each under every key
            it writes.

        Examples:
            >>> CFVariable(units="1").to_attrs()
            {'units': '1'}
        """
        dumped = self.model_dump(
            mode="json",
            exclude_unset=True,
            exclude_computed_fields=True,
        )
        attrs: dict[str, Any] = {}
        for field_name, value in dumped.items():
            for attr_key in type(self).field_keys[field_name]:
                attrs[attr_key] = value
        return attrs

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Keep the fields every object of a join set to the same value.

        A field they set differently describes none of them, so it drops.
        Override this where a field accumulates instead, as STAC provenance does.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            (model the joined result carries, attr keys the dropped fields
            write). A dropped field reads as None.

        Raises:
            ValueError: `models` is empty.

        Examples:
            >>> Packing.merge([Packing(scale_factor=0.0001), Packing()])
            (Packing(scale_factor=None, add_offset=None), {'scale_factor'})
        """
        if not models:
            raise ValueError(f"merging {cls.NAME} needs at least one object")
        present = [model for model in models if model is not None]

        # An object carrying no model agrees with nothing, so every field drops.
        shared: dict[str, Any] = {}
        if len(present) == len(models):
            first, *others = present
            for field_name in cls.model_fields:
                value = getattr(first, field_name)
                if all(
                    attrs_equal(getattr(other, field_name), value)
                    for other in others
                ):
                    shared[field_name] = value

        set_fields: set[str] = set()
        for model in present:
            set_fields.update(model.model_fields_set)

        dropped_fields = sorted(set_fields - shared.keys())

        merged: dict[str, Any] = {}
        # A dropped field is set to None, clearing the key a model had written.
        for field_name in dropped_fields:
            if type(None) in get_args(cls.model_fields[field_name].annotation):
                merged[field_name] = None
        # A field no object set stays unset, so the result clears nothing.
        for field_name, value in shared.items():
            if field_name in set_fields:
                merged[field_name] = value
        dropped_keys: set[str] = set()
        for field_name in dropped_fields:
            dropped_keys.update(cls.field_keys[field_name])
        return cls(**merged), dropped_keys


def parse_field_value(model: type[AttrsModel], field: str, value: object) -> Any:
    """Convert one stored attr to the Python type declared by one model field.

    Some callers have only one field rather than enough values to construct the
    whole model. This applies that field's type annotation, including annotated
    validators, without running the model's decorator or model-level validators.

    Args:
        model: Attrs model declaring the field.
        field: Declared field name.
        value: Stored attr value.

    Returns:
        Parsed field value.

    Raises:
        ValidationError: The value does not satisfy the field.

    Examples:
        >>> parse_field_value(Nodata, "fill_value", "0")
        0
    """
    return model._field_parsers[field].validate_python(value)


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
            return _MODEL_TYPES[model]
        except KeyError:
            raise KeyError(f"no attrs model is registered as {model!r}") from None
    model_name = getattr(model, "NAME", None)
    if model_name is None or _MODEL_TYPES.get(model_name) is not model:
        raise TypeError(f"{model!r} is not a registered concrete AttrsModel")
    return model


def attrs_equal(left: object, right: object) -> bool:
    """Compare values stored in xarray metadata without array broadcasting.

    Array-like values must have the same shape and values. NaNs compare as
    equal. Mappings must contain the same keys and recursively equal values.
    A missing metadata value (`None`) only equals another `None`.

    Args:
        left: First attrs value.
        right: Second attrs value.

    Returns:
        True when both metadata values represent the same value.

    Examples:
        >>> attrs_equal(float("nan"), float("nan"))
        True
    """
    if left is None or right is None:
        return left is right

    if isinstance(left, Mapping):
        if not isinstance(right, Mapping):
            return False
        if left.keys() != right.keys():
            return False
        for key, value in left.items():
            if not attrs_equal(value, right[key]):
                return False
        return True

    if isinstance(right, Mapping):
        return False

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        for left_item, right_item in zip(left, right, strict=True):
            if not attrs_equal(left_item, right_item):
                return False
        return True

    return array_equiv(left, right)


def _field_has_converter(model: type[AttrsModel], field: str) -> bool:
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
    for decorator in (*validators, *serializers):
        if field in decorator.info.fields or "*" in decorator.info.fields:
            return True
    return False
