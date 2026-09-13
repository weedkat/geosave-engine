"""What a label raster's pixel values mean, and how they colour."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

from pydantic import field_validator, model_validator

from geosave_engine.geodata.attrs.model import AttrsModel
from geosave_engine.utils.colorize import Palette


class Legend(AttrsModel):
    """What a label variable's pixel values mean, and how they colour.

    Keyed to pixel values, so it survives every operation that leaves the
    values alone. `flag_values`/`flag_meanings` mirror `class_map` in CF form
    and are derived, not set — only `class_map` makes a legend.

    Args:
        class_map: `{pixel value: class name}` for a label variable. Class
            names carry no whitespace.
        color_map: `{pixel value: hex or RGB}` for a label variable.
        flag_values: Pixel values ascending, mirroring `class_map` keys. Set
            directly only to check consistency against a foreign CF file;
            it never populates `class_map`.
        flag_meanings: Space-separated class names in `flag_values` order,
            mirroring `class_map` values. Same caveat as `flag_values`.

    Raises:
        ValueError: A class name carries whitespace, or `flag_values` and
            `flag_meanings` disagree with `class_map` or each other.

    Examples:
        >>> ds.gs.rebase(
        ...     Legend(class_map={0: "bg", 1: "palm"}), target="labels"
        ... )
    """

    NAME: ClassVar[str] = "legend"

    class_map: dict[int, str] | None = None
    color_map: Palette | None = None
    flag_values: list[int] | None = None
    flag_meanings: str | None = None

    @field_validator("class_map")
    @classmethod
    def _reject_whitespace_in_names(
        cls, value: dict[int, str] | None
    ) -> dict[int, str] | None:
        """Refuse class names carrying whitespace.

        Args:
            value: Proposed `class_map`.

        Returns:
            `value` unchanged.

        Raises:
            ValueError: A class name is empty or contains whitespace, which
                would split into stray `flag_meanings` tokens.
        """
        if value is None:
            return None
        bad = sorted(name for name in value.values() if name.split() != [name])
        if bad:
            raise ValueError(
                f"class names {bad} are empty or carry whitespace; "
                f"flag_meanings tokens are single words, so use '_'"
            )
        return value

    @model_validator(mode="after")
    def _sync_flags(self) -> Self:
        """Derive the CF flag attrs from `class_map`, checking any given flags agree.

        Never the reverse: flags set directly are checked for internal
        consistency, but never populate `class_map`.

        Returns:
            The model with `flag_values`/`flag_meanings` derived from
            `class_map` when it is set, or the given flags checked for
            internal consistency when it is not.

        Raises:
            ValueError: The flag pair disagrees with `class_map`, only one of
                the pair is set, their lengths differ, or `flag_values` is not
                ascending and unique.
        """
        if self.class_map is not None:
            values = sorted(self.class_map)
            meanings = " ".join(self.class_map[value] for value in values)
            if self.flag_values not in (None, values):
                raise ValueError(
                    f"flag_values {self.flag_values} disagree with class_map "
                    f"keys {values}"
                )
            if self.flag_meanings not in (None, meanings):
                raise ValueError(
                    f"flag_meanings {self.flag_meanings!r} disagree with "
                    f"class_map names {meanings!r}"
                )
            if self.flag_values != values:
                self.flag_values = values
            if self.flag_meanings != meanings:
                self.flag_meanings = meanings
            return self

        if self.flag_values is None and self.flag_meanings is None:
            return self
        if self.flag_values is None or self.flag_meanings is None:
            raise ValueError(
                "flag_values and flag_meanings are set together or not at all"
            )
        if len(self.flag_meanings.split()) != len(self.flag_values):
            raise ValueError(
                f"flag_values has {len(self.flag_values)} entries but "
                f"flag_meanings has {len(self.flag_meanings.split())}"
            )
        if sorted(set(self.flag_values)) != self.flag_values:
            raise ValueError(
                f"flag_values {self.flag_values} are not ascending and unique"
            )
        return self

    @classmethod
    def merge(cls, models: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Merge a legend, refusing a different class map.

        Args:
            models: This model from each joined object, in call order, at
                least one, None where an object carried none.

        Returns:
            Merged model, and the attr keys it could not keep.

        Raises:
            TypeError: An object carries a different model.
            ValueError: `class_map` disagrees or `models` is empty.
        """
        return cls._merge_fields(models, must_agree=("class_map",))
