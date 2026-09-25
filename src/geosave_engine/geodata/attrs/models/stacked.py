"""Metadata needed to split a stacked DataArray back into variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Self

import numpy as np
from pydantic import BeforeValidator, ConfigDict

from geosave_engine.geodata.attrs.model import AttrsModel
from geosave_engine.geodata.attrs.namespace import AttrsNamespace
from geosave_engine.geodata.attrs.validate import parse_collection_text

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


class StackedAttrs(AttrsModel):
    """Preserve Dataset and variable attrs while variables share one array.

    Xarray stacks several Dataset variables into one DataArray, which has only
    one attrs mapping. This model records the mappings that no longer have a
    native place and lets `GeoArray.to_raster` restore them.

    Args:
        variable_attrs: Variable name mapped to the attrs it carried.
        dataset_attrs: Dataset values temporarily shadowed by shared variable
            attrs lifted onto the array.

    Examples:
        >>> stacked = ds.gs.to_array()
        >>> stacked.gs.attrs.coords["band"].get(StackedAttrs).variable_attrs
        {'B04': {'units': '1', 'scale_factor': 0.0001}, 'B03': {'units': '1'}}
    """

    NAME: ClassVar[str] = "stacked"
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
    def from_variables(
        cls,
        variables: Mapping[str, Mapping[str, object]],
        *,
        dataset_attrs: Mapping[str, object] | None = None,
    ) -> Self:
        """Capture variable attrs in values supported by persistent stores.

        Args:
            variables: Variable name mapped to its flat attrs.
            dataset_attrs: Dataset attrs. Only values shadowed by shared
                variable attrs are captured; unrelated values stay on the array.

        Returns:
            Captured attrs ready to attach to the stacked array.

        Raises:
            TypeError: An attr value has no JSON representation.
        """
        recorded = cls(variable_attrs=dict(variables))
        if dataset_attrs is not None:
            recorded.dataset_attrs = {
                key: dataset_attrs[key]
                for key in recorded.shared().to_attrs()
                if key in dataset_attrs
            }
        return cls.model_validate(
            recorded.model_dump(mode="json", fallback=_numpy_value)
        )

    def shared(self) -> AttrsNamespace:
        """Return attrs that every captured variable carries identically."""
        shared, _ = AttrsNamespace.merge(
            [
                AttrsNamespace.from_attrs(held)
                for held in (self.variable_attrs or {}).values()
            ]
        )
        return shared

    def restore(self, variables: Collection[str]) -> dict[str, dict[str, object]]:
        """Return captured attrs for variables still present on the array.

        Args:
            variables: Variable labels retained on the stacked axis.

        Returns:
            Retained variable names mapped to their original flat attrs.
        """
        if self.variable_attrs is None:
            return {}
        return {
            label: dict(carried)
            for label, carried in self.variable_attrs.items()
            if label in variables
        }

    def restore_root(self, current: Mapping[str, object]) -> dict[str, object]:
        """Restore shadowed Dataset attrs while retaining unrelated edits."""
        lifted = self.shared().to_attrs()
        return {
            **(self.dataset_attrs or {}),
            **{key: value for key, value in current.items() if key not in lifted},
        }


def _numpy_value(value: object) -> object:
    """Convert numpy scalars and arrays without turning NaN into absence."""
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")
