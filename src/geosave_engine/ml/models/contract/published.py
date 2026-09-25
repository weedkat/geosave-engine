"""Build-time contract: attributes one stage offers the stages built after it."""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Mapping
import typing
from typing import Annotated, TypeVar, get_args, get_origin, get_type_hints

import torch.nn as nn

_PUBLISHED = "geosave.published"

_T = TypeVar("_T")

Published = Annotated[_T, _PUBLISHED]
"""Mark a class attribute as readable by later stages in `build_model`.

Annotate at class level; the runtime value stays a plain ``T``, and a stage
built later whose constructor takes a parameter of the same name and type
receives it automatically.

Examples:
    >>> class DINOv3(nn.Module):
    ...     pyramid_channels: Published[list[int]]
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.pyramid_channels = [768] * 4
"""


def accepts(declared: type, wanted: object) -> bool:
    """Check a published type against a constructor parameter annotation.

    Args:
        declared: Type marked with Published on the source attribute.
        wanted: Receiving parameter's annotation, optionally including None.

    Returns:
        Whether the types match after removing None from the receiving type.
    """
    if get_origin(wanted) not in (types.UnionType, typing.Union):
        return wanted == declared
    members = set(get_args(wanted)) - {type(None)}
    if get_origin(declared) in (types.UnionType, typing.Union):
        return members == set(get_args(declared))
    return members == {declared}


def published_attrs(cls: type[nn.Module]) -> dict[str, type]:
    """Attributes `cls` and its bases mark with `Published`.

    Args:
        cls: Stage class to inspect.

    Returns:
        Attribute name mapped to its declared type, inherited marks included.
    """
    hints = get_type_hints(cls, include_extras=True)
    return {
        name: get_args(hint)[0]
        for name, hint in hints.items()
        if get_origin(hint) is Annotated and _PUBLISHED in get_args(hint)[1:]
    }


def published_kwargs(
    factory: Callable[..., nn.Module],
    built: Mapping[str, nn.Module],
    explicit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fill omitted constructor arguments from earlier stages' Published attributes.

    Args:
        factory: Class or factory to construct.
        built: Earlier stages available as argument sources.
        explicit: Arguments that take precedence over published attributes.

    Returns:
        Explicit arguments plus uniquely matched published values.

    Raises:
        TypeError: An omitted argument has ambiguous or incompatible sources.
    """
    sources: dict[str, list[tuple[str, type]]] = {}
    for stage, module in built.items():
        for attr, declared in published_attrs(type(module)).items():
            sources.setdefault(attr, []).append((stage, declared))

    constructor = factory.__init__ if isinstance(factory, type) else factory
    hints = get_type_hints(constructor)
    resolved = dict(explicit or {})
    for param in inspect.signature(factory).parameters:
        if param in resolved:
            continue
        candidates = sources.get(param)
        if candidates is None:
            continue
        if len(candidates) > 1:
            raise TypeError(
                f"Argument {param!r} is published by {[stage for stage, _ in candidates]}; "
                "set it explicitly in init_args"
            )
        stage, declared = candidates[0]
        if not accepts(declared, hints.get(param, declared)):
            raise TypeError(
                f"Argument {param!r} is {hints.get(param)}, but "
                f"{type(built[stage]).__name__} publishes {declared}"
            )
        resolved[param] = getattr(built[stage], param)
    return resolved
