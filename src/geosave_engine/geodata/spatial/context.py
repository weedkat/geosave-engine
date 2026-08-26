"""Model context: one window's precomputed model inputs, and how they render."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from .anchor import GeoAnchor

# One window's precomputed model inputs, held as the encoder produced them.
ModelContext: TypeAlias = dict[str, Any]
# One window's model inputs, off its anchor alone — whose header carries bands, times and nodata.
ContextFn: TypeAlias = "Callable[[GeoAnchor], Mapping[str, object] | None]"

# What one walk of a context is for: reject what a model couldn't read, or render for a runtime.
Target: TypeAlias = Literal["check", "numpy", "tensor"]

# dtype kinds that both survive JSON and fit a tensor: bool, signed/unsigned int, float. Not complex/text/object.
_STORABLE_KINDS = "biuf"


def _as_array(value: object, path: str) -> np.ndarray:
    """One array-like leaf as numpy, rejecting anything that can't round-trip.

    Args:
        value: Array, list, tuple or scalar to convert.
        path: Dotted key path, for the error message.

    Returns:
        The value as an array of a storable dtype.

    Raises:
        ValueError: The value isn't rectangular, or its dtype is complex,
            text, object or datetime rather than numeric or boolean.
    """
    try:
        array = np.asarray(value)
    except Exception as error:
        raise ValueError(f"model_context {path} isn't a rectangular array: {error}") from error
    if array.dtype.kind not in _STORABLE_KINDS:
        raise ValueError(
            f"model_context {path} has dtype {array.dtype}, which cannot both serialize and become "
            "a tensor — a context array must be numeric or boolean; pass text as a plain string, "
            "not a sequence of them"
        )
    return array


def _render(value: object, target: Target, path: str) -> Any:
    """Walk one context value for `target`, rejecting anything a model couldn't read.

    Every target walks the same shape and accepts the same leaves, so a
    value that passes `"check"` is one `"numpy"` and `"tensor"` can render.
    Tensors are an output form only — a stored value is never one.

    Args:
        value: Context value to walk.
        target: `"check"` returns leaves untouched, `"numpy"` returns
            arrays, `"tensor"` returns tensors.
        path: Dotted key path to `value`, for error messages.

    Returns:
        The rebuilt value. Nested mappings stay nested; strings and None
        pass through whatever the target.

    Raises:
        ValueError: A nested mapping has a non-string key, or a leaf is
            neither a scalar nor an array-like of numbers.
    """
    if isinstance(value, Mapping):
        rendered: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"model_context {path} has non-string key {key!r}")
            rendered[key] = _render(item, target, f"{path}.{key}")
        return rendered

    if value is None or isinstance(value, str):
        return value

    if not isinstance(value, (bool, int, float, np.generic, np.ndarray, list, tuple)):
        # a tensor is the likely mistake here, so name the fix rather than only the rule
        raise ValueError(
            f"model_context {path} is a {type(value).__name__}; expected an array, scalar, string, "
            "None, or a nested mapping of those. A tensor cannot be stored — build the value with "
            "numpy and its dtype, and read it back with to_sample(); that is what tensorizes it"
        )

    # the conversion is also the check, so "check" runs it and keeps the value as the encoder made it
    array = _as_array(value, path)
    if target == "check":
        return value
    if target == "numpy":
        return array

    import torch

    # a store decodes from a read-only buffer; torch cannot own one, so copy instead
    return torch.as_tensor(array if array.flags.writeable else array.copy())


def validate_context(values: Mapping[str, object]) -> ModelContext:
    """Shallow-copy model context, checking keys and value kinds only.

    Every value must be JSON-serializable, so a context survives being written
    beside its tile. Values are otherwise kept exactly as produced, dtype
    included — `to_sample` hands back the dtype the array was built with.

    Args:
        values: Model inputs for one window.

    Returns:
        A copy of `values`, unconverted.

    Raises:
        ValueError: A key isn't a string, or a value is neither array-like,
            a scalar, a string, None, nor a nested mapping of those. A
            tensor is rejected — it cannot serialize.

    Examples:
        >>> validate_context({"latlon": np.array([52.0, 13.0], dtype="float32")})
        {'latlon': array([52., 13.], dtype=float32)}
    """
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError(f"model_context has non-string key {key!r}")
        _render(value, "check", key)
    return dict(values)


def numpy_context(context: ModelContext | None) -> dict[str, Any]:
    """Every value in a context rendered as numpy.

    Args:
        context: Stored context, or None for a window that carries none.

    Returns:
        `{key: array | str | None}`, nested mappings kept nested. Empty when
        `context` is None.
    """
    return {key: _render(value, "numpy", key) for key, value in (context or {}).items()}


def tensor_context(context: ModelContext | None) -> dict[str, Any]:
    """Every value in a context rendered as tensors.

    Args:
        context: Stored context, or None for a window that carries none.

    Returns:
        `{key: tensor | str | None}`, nested mappings kept nested. Empty when
        `context` is None.
    """
    return {key: _render(value, "tensor", key) for key, value in (context or {}).items()}
