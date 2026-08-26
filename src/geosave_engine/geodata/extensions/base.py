"""GeoExtension: namespaced, self-registering pydantic schema. See GeoExtension for details."""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Self

import orjson
from pydantic import BaseModel, ConfigDict

from geosave_engine.geodata.errors import UnknownExtensionError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import xarray as xr

_REGISTRY: dict[str, type[GeoExtension]] = {}
_REGISTRY_VIEW = MappingProxyType(_REGISTRY)

# Namespaces that can never be extensions: each names a real GeoAnchor/GeoRaster field.
_RESERVED_NAMESPACES = frozenset({"data", "geobox", "nodata", "vector"})


class GeoExtension(BaseModel):
    """One namespaced group of fields a raster/tile/anchor can carry.

    Subclassing registers it — no decorator, no import needed by the code
    that reads it back. Every one lands on disk under its own namespace
    key (see `GeoHeader`).

    Args:
        **fields: Whatever the subclass declares. All should default to
            None or a literal so a bare `Cls()` is always constructible.

    Examples:
        >>> class RenderHints(GeoExtension):
        ...     NAMESPACE: ClassVar[str] = "render"
        ...     class_map: dict[int, str] | None = None
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    NAMESPACE: ClassVar[str] = ""
    # False keeps this namespace out of GeoHeader.rebase()'s generic merge path — only
    # the operation that legitimately produces it may set it (see TimeSpec).
    SETTABLE: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register cls under its own NAMESPACE. A subclass leaving NAMESPACE empty is abstract, skipped.

        Raises:
            ValueError: `NAMESPACE` is already registered by another class,
                or names one of `rebase`'s own parameters.
        """
        super().__init_subclass__(**kwargs)

        # NAMESPACE read off cls's own dict, not inherited — an intermediate base stays abstract.
        namespace = cls.__dict__.get("NAMESPACE", "")
        if not namespace:
            return
        if namespace in _RESERVED_NAMESPACES:
            raise ValueError(
                f"NAMESPACE {namespace!r} is reserved — rebase() binds it to its own parameter, "
                f"so the extension would be unreachable. Reserved: {sorted(_RESERVED_NAMESPACES)}"
            )

        # Same qualname means a module reload redefining the same class, not a real clash.
        claimed = _REGISTRY.get(namespace)
        if claimed is not None and claimed.__qualname__ != cls.__qualname__:
            raise ValueError(f"NAMESPACE {namespace!r} already registered by {claimed.__qualname__}")

        _REGISTRY[namespace] = cls

    @classmethod
    def registry(cls) -> Mapping[str, type[GeoExtension]]:
        """Every registered extension, `{namespace: class}`.

        Returns:
            Read-only live view — importing a module that defines an
            extension adds to it, so a missing namespace means unimported.
        """
        return _REGISTRY_VIEW

    @classmethod
    def combine(cls, values: Sequence[GeoExtension | None]) -> Self | None:
        """Merge this namespace's values from arrays being composed into one.

        Called by every op that folds several arrays into one — `concat`,
        `merge_spatial`. N-ary, not pairwise: a None slot says that input
        carried nothing. The default keeps values that agree, rejects the rest.

        Args:
            values: One value per input array, in composition order. None
                means that input does not carry this namespace. Never empty.

        Returns:
            The merged value, or None to leave the namespace off the result.

        Raises:
            ValueError: The values disagree and this extension declares no
                rule for merging them.
        """
        first = values[0]
        if isinstance(first, cls) and all(value == first for value in values[1:]):
            return first
        raise ValueError(
            f"cannot compose arrays whose {cls.NAMESPACE!r} extensions differ — "
            f"{cls.__name__} declares no merge rule, so align or clear it first"
        )

    def reconcile(self, data: xr.DataArray) -> Self | None:
        """Refresh this value against the array it's attached to.

        Called once per namespace whenever a `GeoHeader` is built with a
        `data` array given. Idempotent. The default keeps the value as it
        stands — override to repair, drop, or re-read one the data owns.

        Args:
            data: This array's own canonical pixel data.

        Returns:
            This value, a corrected copy, or None to drop the namespace.
        """
        return self

    def encode(self) -> dict[str, Any] | None:
        """Fields ready for one store's attrs, this namespace's own key.

        The default is a plain JSON-mode dump. Override for a namespace
        whose fields need something other than pydantic's own JSON
        conversion.

        Returns:
            `{field: value}`, JSON-compatible, `None` fields excluded; or
            None if every field is empty, which omits the namespace entirely.
        """
        dumped = self.model_dump(mode="json", exclude_none=True)
        return dumped or None

    @classmethod
    def decode(cls, value: Any) -> Self:
        """Parse one stored namespace value back into this extension.

        The single entry for anything not already an instance — stored attrs
        and `rebase`'s merged mappings both arrive here, so an override is
        honoured on every path. Accepts a JSON string or a decoded mapping.

        Args:
            value: This namespace's stored value, either form `encode` writes.

        Returns:
            New instance.

        Raises:
            ValueError: `value` fails this extension's own field validation.
        """
        return cls.model_validate(orjson.loads(value) if isinstance(value, str) else value)

    @classmethod
    def lookup(cls, namespace: str) -> type[GeoExtension]:
        """Registered class for one namespace.

        Args:
            namespace: Namespace string, e.g. `"render"`.

        Returns:
            The registered subclass.

        Raises:
            UnknownExtensionError: nothing registered under `namespace`.
        """
        try:
            return _REGISTRY[namespace]
        except KeyError:
            raise UnknownExtensionError(
                f"No extension registered for namespace {namespace!r} — import its module first. "
                f"Registered: {sorted(_REGISTRY)}"
            ) from None
