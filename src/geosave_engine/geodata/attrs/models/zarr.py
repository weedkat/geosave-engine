"""What a Zarr store cannot say about itself."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

from geosave_engine.geodata.attrs.model import AttrsModel


class ZarrOrder(AttrsModel):
    """State the order a Zarr store's variables were written in.

    A Zarr group holds its members as a mapping, so the store records no order
    and two reads of one store can disagree. The order is written here instead,
    and `utils.io.zarr.read` restores it.

    Args:
        zarr_variable_order: Data variable names, in the order they were
            written.

    Examples:
        >>> ds.gs.attrs.root.get(ZarrOrder).zarr_variable_order
        ('B04', 'B03', 'B02')
    """

    NAME: ClassVar[str] = "zarr"

    zarr_variable_order: tuple[str, ...] | None = None

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge written orders, dropping disagreements.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            Model carrying the order every object agreed on, and the attr keys
            it could not keep.

        Raises:
            TypeError: An object carries a different model.
            ValueError: `models` is empty.
        """
        return cls._merge_fields(models, must_agree=())
