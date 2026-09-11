"""Read typed attrs off an xarray object, and write them back."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, overload

import xarray as xr

from geosave_engine.geodata.errors import DroppedAttrsWarning

from .header import AttrsHeader
from .namespace import AttrsNamespace
from .model import AttrsModel, resolve_model

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

type XarrayObject = xr.Dataset | xr.DataArray | xr.DataTree


def read(obj: XarrayObject) -> AttrsHeader:
    """Read every registered model an object states.

    A model is stated where at least one of its keys is present, and
    `model_fields_set` says which keys the object actually carried. A DataTree
    node is read on its own, its children untouched.

    Args:
        obj: Dataset, DataArray, or DataTree to read.

    Returns:
        Header holding the models and foreign keys the object carries.

    Raises:
        ValidationError: A stated value does not satisfy its field.

    Examples:
        >>> read(ds).data_vars["B04"].get("cf").units
        '1'
    """
    coord_names = _coord_names(obj)
    var_names = _var_names(obj)
    carried = _variables(obj)
    variables = {
        name: AttrsNamespace.from_attrs(carried[name].attrs)
        for name in (*coord_names, *var_names)
    }
    return AttrsHeader(
        root=AttrsNamespace.from_attrs(obj.attrs),
        variables=variables,
        coord_names=coord_names,
        var_names=var_names,
    )


def combine(objects: Sequence[XarrayObject]) -> AttrsHeader:
    """Work out the attrs a join of these objects should end up with.

    Each model decides for itself what survives, through its own `combine`.
    Variables and coordinates are matched by name, and an object that does not
    have a name has no say about it.

    Args:
        objects: The objects being joined, at least one.

    Returns:
        Header to stamp onto the joined result.

    Raises:
        ValueError: `objects` is empty, or a model refused a value the objects
            disagreed on.

    Warns:
        DroppedAttrsWarning: A model or foreign attr dropped because the
            objects did not state it alike.

    Examples:
        >>> stamp(xr.concat(rasters, dim="time"), combine(rasters))
    """
    if not objects:
        raise ValueError("combining attrs needs at least one object")
    header, dropped = AttrsHeader.combine([read(obj) for obj in objects])
    if dropped:
        named = sorted(str(attr) for attr in dropped)
        warnings.warn(
            f"joining drops attrs the objects did not state alike: {named}",
            DroppedAttrsWarning,
            stacklevel=2,
        )
    return header


@overload
def rebase[T: XarrayObject](
    obj: T,
    *models: AttrsModel,
    target: str | Sequence[str] | None = None,
    inplace: Literal[False] = False,
    **model_kwargs: Mapping[str, object] | None,
) -> T: ...


@overload
def rebase(
    obj: XarrayObject,
    *models: AttrsModel,
    target: str | Sequence[str] | None = None,
    inplace: Literal[True],
    **model_kwargs: Mapping[str, object] | None,
) -> None: ...


def rebase[T: XarrayObject](
    obj: T,
    *models: AttrsModel,
    target: str | Sequence[str] | None = None,
    inplace: bool = False,
    **model_kwargs: Mapping[str, object] | None,
) -> T | None:
    """Write models onto an object, replacing the keys they own.

    Models are applied in order, keyword models last, so where two write one
    key the last wins. A field stating None removes its key, a keyword stating
    None removes every key its model writes, and other keys are left alone.

    Args:
        obj: Dataset, DataArray, or DataTree to write onto. A DataTree node
            is written on its own, its children untouched.
        *models: Model instances to apply to `target`.
        target: Variable or coordinate name the models describe, or several
            of them. None writes to the object's own attrs, which no name can
            address since a variable cannot be called None.
        inplace: Write into `obj` rather than returning a new object.
        **model_kwargs: Model name mapped to its field values, e.g.
            ``legend={"class_map": {...}}``, or to None to drop that model.

    Returns:
        New object carrying the models, or None when `inplace` is set.

    Raises:
        KeyError: A keyword names no registered model.
        ValueError: `target` names neither a variable nor a coordinate of
            `obj`.
        ValidationError: A supplied value does not satisfy its field.

    Examples:
        >>> rebase(ds, Packing(fill_value=0), target=ds.gs.variables)
        >>> rebase(ds, acdd={"title": "Sentinel-2 Level-2A"})
        >>> rebase(stretched, timespec=None)  # the recorded bucketing no longer holds
    """
    wrong_models = sorted(
        {type(model).__name__ for model in models if not isinstance(model, AttrsModel)}
    )
    if wrong_models:
        raise TypeError(f"rebase needs AttrsModel instances, got {wrong_models}")

    result = obj if inplace else obj.copy(deep=False)  # type: ignore

    models_to_apply = list(models)
    models_to_remove: list[type[AttrsModel]] = []
    for name, values in model_kwargs.items():
        model = resolve_model(name)
        if values is None:
            models_to_remove.append(model)
        else:
            models_to_apply.append(model(**values))
    # None means the object's own attrs, which no variable name can collide with.
    if target is None or isinstance(target, str):
        target_names: Sequence[str | None] = (target,)
    else:
        target_names = target

    for target_name in target_names:
        resolved = _resolve_target(result, target_name)
        attrs = dict(resolved.attrs)
        for removed_model in models_to_remove:
            for attr_key in removed_model.attr_keys.values():
                attrs.pop(attr_key, None)
        for applied_model in models_to_apply:
            for key, value in applied_model.to_attrs().items():
                if value is None:
                    attrs.pop(key, None)
                else:
                    attrs[key] = value
        resolved.attrs = attrs

    return None if inplace else result


@overload
def stamp[T: XarrayObject](
    obj: T, header: AttrsHeader, *, inplace: Literal[False] = False
) -> T: ...


@overload
def stamp(
    obj: XarrayObject, header: AttrsHeader, *, inplace: Literal[True]
) -> None: ...


def stamp[T: XarrayObject](
    obj: T, header: AttrsHeader, *, inplace: bool = False
) -> T | None:
    """Replace the represented attr namespaces with an attrs header.

    A header is a snapshot, so the object's own attrs and each variable's
    attrs it represents are replaced as a whole. Variables absent from the
    header are left alone.

    Args:
        obj: Dataset, DataArray, or DataTree to stamp. A DataTree node is
            stamped on its own, its children untouched.
        header: Detached attrs snapshot to write.
        inplace: Write into `obj` rather than returning a new object.

    Returns:
        New object carrying the header, or None when `inplace` is set.

    Raises:
        KeyError: `header` names an unregistered model.
        TypeError: A header model does not match its registered name.
        ValueError: A foreign key collides with a registered attr key, or a
            header target is absent from `obj`.
    """
    result = obj if inplace else obj.copy(deep=False)  # type: ignore

    # Resolve every target before writing so an absent one leaves obj untouched.
    targets = [
        (_resolve_target(result, name), namespace)
        for name, namespace in header.variables.items()
    ]

    result.attrs = header.root.to_attrs()
    for resolved, namespace in targets:
        resolved.attrs = namespace.to_attrs()
    return None if inplace else result


def _coord_names(obj: XarrayObject) -> frozenset[str]:
    """Return the coordinate names an object carries."""
    return frozenset(str(name) for name in obj.coords)


def _var_names(obj: XarrayObject) -> frozenset[str]:
    """Return the data-variable names an object carries, none for a DataArray."""
    if isinstance(obj, xr.DataArray):
        return frozenset()
    return frozenset(str(name) for name in obj.data_vars)


def _variables(obj: XarrayObject) -> Mapping[Hashable, xr.Variable]:
    """Return the Variables an object carries, building no DataArray per name."""
    if isinstance(obj, xr.DataArray):
        return obj.coords.variables
    return obj.variables


def _resolve_target(obj: XarrayObject, target: str | None) -> XarrayObject:
    """Return the xarray object holding `target`'s attrs, `obj` itself for None."""
    if target is None:
        return obj
    carried = _coord_names(obj) | _var_names(obj)
    if target in carried:
        return obj[target]
    raise ValueError(
        f"{target!r} is neither a variable nor a coordinate of this "
        f"{type(obj).__name__}; it carries {sorted(carried)}"
    )
