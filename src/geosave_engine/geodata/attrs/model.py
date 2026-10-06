"""Typed metadata models carried in flat xarray attrs mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, ClassVar, Self

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
        NAME: Stable name, unique across `MODELS`, used as `rebase`'s keyword.
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

    NAME: ClassVar[str]
    field_keys: ClassVar[Mapping[str, tuple[str, ...]]] = {}

    @classmethod
    def attr_keys(cls, field_name: str | None = None) -> tuple[str, ...]:
        """Return the attr keys one field writes, or every key this model writes."""
        if field_name is not None:
            return cls.field_keys.get(field_name, (field_name,))
        return tuple(key for name in cls.model_fields for key in cls.attr_keys(name))

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
            spellings = [key for key in cls.attr_keys(field_name) if key in attrs]
            if not spellings:
                continue
            if len(spellings) > 1:
                # One store may hold a spelling as text and another as a number.
                typed = [
                    parse_field_value(cls, field_name, attrs[key]) for key in spellings
                ]
                for spelling, other in zip(spellings[1:], typed[1:], strict=True):
                    if not attrs_equal(other, typed[0]):
                        raise ValueError(
                            f"{spellings[0]!r} is {typed[0]!r} but {spelling!r} is "
                            f"{other!r}; they spell one {cls.__name__}.{field_name}, "
                            f"so set one of them"
                        )
            field_values[field_name] = attrs[spellings[0]]
        return cls(**field_values) if field_values else None

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
        return {
            key: value
            for field_name, value in self.model_dump(
                mode="json",
                exclude_unset=True,
                exclude_computed_fields=True,
                fallback=_json_value,
            ).items()
            if value is not None
            for key in self.attr_keys(field_name)
        }

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Keep the fields every object of a join set to the same value.

        A field they set differently drops; one marked `MUST_AGREE` refuses,
        since it says what the joined pixels mean. Override this where a
        field accumulates instead, as STAC provenance does.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            (model the joined result carries, attr keys the dropped fields
            write). A dropped field reads as None.

        Raises:
            ValueError: `models` is empty, or the objects carry a field marked
                `MUST_AGREE` differently, including one carrying it and another
                not.

        Examples:
            >>> CFVariable.merge([CFVariable(long_name="Red"), CFVariable()])
            (CFVariable(standard_name=None, long_name=None, units=None, cell_methods=None), {'long_name'})
        """
        if not models:
            raise ValueError(f"merging {cls.NAME} needs at least one object")

        for name, field_info in cls.model_fields.items():
            if MUST_AGREE in field_info.metadata:
                values = [
                    None if model is None else getattr(model, name) for model in models
                ]
                if not all_equal(values):
                    raise ValueError(
                        f"{cls.NAME}.{name} must agree across the joined objects, "
                        f"but they carry {values}; align it before joining them"
                    )

        fields_set = set().union(*(m.model_fields_set for m in models if m is not None))
        common_fields = {
            name
            for name in cls.model_fields
            # An object carrying no model has nothing in common with the others.
            if None not in models and all_equal([getattr(m, name) for m in models])
        }
        dropped_fields = fields_set - common_fields
        merged = {name: getattr(models[0], name) for name in common_fields & fields_set}
        dropped_keys = {key for name in dropped_fields for key in cls.attr_keys(name)}
        return cls(**merged), dropped_keys


@cache
def _field_adapter(model: type[AttrsModel], field: str) -> TypeAdapter[Any]:
    """Build the validator for one field's annotation, once per field."""
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


def _json_value(value: object) -> object:
    """Convert NumPy scalars and arrays without turning NaN into absence."""
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"attr value {value!r} has no JSON representation")
