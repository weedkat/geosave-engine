"""What a stacked DataArray's one attrs mapping cannot hold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BeforeValidator, ConfigDict

from geosave_engine.geodata.attrs.model import AttrsModel, parse_collection_text

if TYPE_CHECKING:
    from collections.abc import Iterable

    from geosave_engine.geodata.attrs.header import AttrsHeader
    from geosave_engine.geodata.attrs.namespace import AttrsNamespace


class StackedAttrs(AttrsModel):
    """Keep the attrs a stacked array's root cannot hold, on its band axis.

    The root of a stacked array holds what every band shares; this holds the
    rest, so `GeoArray.to_raster` restores each band and the Dataset root.

    Args:
        variable_attrs: Band name mapped to the attrs not every band shares.
        dataset_attrs: Attrs of the Dataset the bands were stacked from.

    Examples:
        >>> stacked = ds.gs.to_array()
        >>> stacked.gs.attrs.coords["band"].get(StackedAttrs).variable_attrs
        {'B04': {'scale_factor': 0.0001}, 'B03': {}}
    """

    model_config = ConfigDict(ser_json_inf_nan="constants")

    variable_attrs: Annotated[
        dict[str, dict[str, object]] | None,
        BeforeValidator(parse_collection_text),
    ] = None
    dataset_attrs: Annotated[
        dict[str, object] | None,
        BeforeValidator(parse_collection_text),
    ] = None

    @classmethod
    def from_header(cls, header: AttrsHeader) -> tuple[AttrsNamespace, Self]:
        """Separate a Dataset header into shared and preserved attrs.

        Only variable namespaces merge. Differing fields remain with their
        variables; the Dataset root stays separate, even for matching keys.

        Args:
            header: Dataset header with at least one data variable.

        Returns:
            Shared variable attrs for the array root and metadata to preserve
            on its band coordinate.

        Raises:
            ValueError: The header has no data variables.
        """
        from geosave_engine.geodata.attrs.namespace import AttrsNamespace

        shared, _ = AttrsNamespace.merge(
            list(header.data_vars.values()), conflicts="drop"
        )
        shared_keys = shared.to_attrs().keys()
        preserved = cls(
            variable_attrs={
                name: {
                    key: value
                    for key, value in namespace.to_attrs().items()
                    if key not in shared_keys
                }
                for name, namespace in header.data_vars.items()
            },
            dataset_attrs=header.root.to_attrs(),
        )
        return shared, preserved

    def to_header(
        self, *, variables: Iterable[str], shared: AttrsNamespace
    ) -> AttrsHeader:
        """Restore Dataset attrs for the variables still present in an array.

        Args:
            variables: Variable names in the restored Dataset, in order.
            shared: Current array attrs, applied to every restored variable.

        Returns:
            Dataset header with its original root and each variable's attrs.
            Current shared attrs override preserved variable attrs.
        """
        from geosave_engine.geodata.attrs.header import AttrsHeader

        own = self.variable_attrs or {}
        common = shared.to_attrs()
        return AttrsHeader.from_attrs(
            root=self.dataset_attrs,
            data_vars={name: own.get(name, {}) | common for name in variables},
        )
