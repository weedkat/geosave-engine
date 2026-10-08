"""Typed metadata models carried in flat xarray attrs mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, ClassVar, Literal, Self

import numpy as np
import orjson
from pydantic import BaseModel, ConfigDict, TypeAdapter
from xarray.core.duck_array_ops import array_equiv


type FlatAttrs = dict[str, Any]  # attr key to stored value, as `.attrs` holds them


class MustAgree:
    """Mark a field that same-named variables joined side by side must agree on."""


MUST_AGREE = MustAgree()


class AttrsModel(BaseModel):
    """One named group of flat attr keys, read and written as typed fields.

    Each field reads and writes its own name in an xarray `.attrs` mapping
    unless `field_keys` declares other keys.

    Attributes:
        field_keys: Field names mapped to the attr keys they write, for fields
            spelled several ways; `Nodata.fill_value` writes
            `("_FillValue", "nodata")`.

    Args:
        **fields: Values for the concrete model's own fields.

    Raises:
        ValidationError: A value does not satisfy its field.

    Examples:
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

    field_keys: ClassVar[Mapping[str, tuple[str, ...]]] = {}

    @classmethod
    @cache
    def attr_keys(cls) -> tuple[str, ...]:
        """Return all stored keys owned by this model, in field order.

        The declared fields and aliases are fixed for each model class, so
        their key list is computed once. This includes keys for unset fields.

        Examples:
            >>> Nodata.attr_keys()
            ('_FillValue', 'nodata')
        """
        keys: list[str] = []
        for name in cls.model_fields:
            keys.extend(cls.keys_for(name))
        return tuple(keys)

    @classmethod
    def keys_for(cls, field: str) -> tuple[str, ...]:
        """Return the stored aliases for one declared model field.

        Args:
            field: Logical Pydantic field name, such as `fill_value`.

        Returns:
            Tuple of stored attr keys. A field without aliases uses its name.

        Raises:
            KeyError: The field is not declared by this model.

        Examples:
            >>> Nodata.keys_for("fill_value")
            ('_FillValue', 'nodata')
        """
        if field not in cls.model_fields:
            raise KeyError(field)
        return cls.field_keys.get(field, (field,))

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> Self | None:
        """Parse this model from a flat attrs mapping.

        Args:
            attrs: Flat attrs mapping, possibly carrying other models' keys.

        Returns:
            The model, or None when the mapping carries none of its keys.

        Raises:
            ValueError: A field's spellings are set to different values.
            ValidationError: A value does not satisfy its field.

        Examples:
            >>> Nodata.from_attrs({"nodata": 0, "units": "1"})
            Nodata(fill_value=0)
        """
        field_values: dict[str, Any] = {}
        for field_name in cls.model_fields:
            spellings = [key for key in cls.keys_for(field_name) if key in attrs]
            if not spellings:
                continue
            first_key = spellings[0]
            value = attrs[first_key]
            if len(spellings) > 1:
                # One store may hold a spelling as text and another as a number.
                parsed = parse_field_value(cls, field_name, value)
                for spelling in spellings[1:]:
                    other = parse_field_value(cls, field_name, attrs[spelling])
                    if not attrs_equal(other, parsed):
                        raise ValueError(
                            f"{first_key!r} is {parsed!r} but {spelling!r} is "
                            f"{other!r}; they spell one {cls.__name__}.{field_name}, "
                            f"so set one of them"
                        )
            field_values[field_name] = _native_number(value)
        return cls.model_validate(field_values) if field_values else None

    def to_attrs(self) -> FlatAttrs:
        """Read this model back as a flat attrs mapping.

        Values come out JSON-native (datetimes as ISO 8601 strings,
        timedeltas as ISO 8601 durations, dict keys as strings, NumPy values
        as Python ones) so every attr stays writable to zarr and netCDF.

        Returns:
            {
                "<attr key>": its value,
            }
            Only fields explicitly set to a value appear, each under every key
            it writes.

        Raises:
            TypeError: A value has no JSON representation.

        Examples:
            >>> CFVariable(units="1").to_attrs()
            {'units': '1'}
        """
        fields = self.model_dump(
            mode="json",
            exclude_unset=True,
            exclude_computed_fields=True,
            fallback=_json_value,
        )
        attrs: FlatAttrs = {}
        for field_name, value in fields.items():
            if value is None:
                continue
            for key in self.keys_for(field_name):
                attrs[key] = value
        return attrs

    @classmethod
    def merge(
        cls,
        models: Sequence[AttrsModel | None],
        *,
        conflicts: Literal["raise", "drop"] = "raise",
    ) -> tuple[Self, set[str]]:
        """Keep the fields every object of a join set to the same value.

        A field they set differently drops; one marked `MUST_AGREE` raises
        unless `conflicts="drop"`. Override this where a field accumulates
        instead, as STAC provenance does.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.
            conflicts: Whether differing `MUST_AGREE` fields raise or drop.

        Returns:
            (model the joined result carries, attr keys the dropped fields
            write). A dropped field reads as None.

        Raises:
            ValueError: `models` is empty, or the objects carry a field marked
                `MUST_AGREE` differently, including one carrying it and another
                not, when `conflicts="raise"`.

        Examples:
            >>> CFVariable.merge([CFVariable(long_name="Red"), CFVariable()])
            (CFVariable(standard_name=None, long_name=None, units=None, cell_methods=None), {'long_name'})
        """
        if not models:
            raise ValueError(f"merging {cls.__name__} needs at least one object")

        fields_set: set[str] = set()
        missing_model = False
        for model in models:
            if model is None:
                missing_model = True
            else:
                fields_set.update(model.model_fields_set)

        merged: dict[str, Any] = {}
        dropped_keys: set[str] = set()
        for name, field_info in cls.model_fields.items():
            values: list[Any] = []
            for model in models:
                values.append(None if model is None else getattr(model, name))
            agreed = all_equal(values)
            if (
                conflicts == "raise"
                and MUST_AGREE in field_info.metadata
                and not agreed
            ):
                raise ValueError(
                    f"{cls.__name__}.{name} must agree across the joined objects, "
                    f"but they carry {values}; align it before joining them"
                )
            if name not in fields_set:
                continue
            if agreed and not missing_model:
                merged[name] = values[0]
            else:
                dropped_keys.update(cls.keys_for(name))
        return cls(**merged), dropped_keys


@cache
def _field_adapter(model: type[AttrsModel], field: str) -> TypeAdapter[Any]:
    """Reuse one Pydantic validation schema per model class and field name.

    Creating a TypeAdapter builds a validator and serializer. Caching the
    adapter avoids rebuilding them on each parse; values are still validated
    on every call.
    """
    return TypeAdapter(model.model_fields[field].rebuild_annotation())


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
    return _field_adapter(model, field).validate_python(value)


def all_equal(values: Sequence[object]) -> bool:
    """Return whether every value equals the first, by `attrs_equal`."""
    return all(attrs_equal(value, values[0]) for value in values[1:])


def common_attrs(mappings: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the keys every mapping carries with one value.

    Args:
        mappings: Flat attrs mappings, at least one.

    Returns:
        {
            "<attr key>": the value every mapping carries under it,
        }

    Examples:
        >>> common_attrs([{"units": "1", "nodata": 0}, {"units": "1"}])
        {'units': '1'}
    """
    first, *rest = mappings
    return {
        key: value
        for key, value in first.items()
        if all(key in other and attrs_equal(other[key], value) for other in rest)
    }


def parse_collection_text(value: object) -> object:
    """Decode JSON text while leaving native values and other text unchanged.

    GDAL tags are text-only, so lists and mappings come back as JSON text where
    Zarr and netCDF return native collections; fields run this before their own
    validation so both read the same.

    Args:
        value: Stored attr value.

    Returns:
        Decoded JSON when `value` is valid JSON text; otherwise `value` itself.
    """
    if not isinstance(value, str):
        return value
    try:
        return orjson.loads(value)
    except orjson.JSONDecodeError:
        return value


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

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(attrs_equal(value, right[key]) for key, value in left.items())
        )

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            attrs_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )

    return array_equiv(left, right)


def _native_number(value: object) -> object:
    """Read a NumPy integer or boolean a file hands back as a Python one.

    Pydantic reads a NumPy integer offered to an `int | float` field as a
    float, so a `uint16` fill of `0` would otherwise become `0.0`.

    Args:
        value: Stored attr value.

    Returns:
        An `int` or `bool` for a NumPy integer or boolean. Any other value is
        returned as it came.
    """
    if isinstance(value, np.integer | np.bool_):
        return value.item()
    return value


def _json_value(value: object) -> object:
    """Convert NumPy scalars and arrays without turning NaN into absence."""
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"attr value {value!r} has no JSON representation")
