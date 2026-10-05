"""What a label variable's pixel values mean, and how they colour."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar, NamedTuple, Self

from pydantic import BeforeValidator, model_validator

from geosave_engine.geodata.attrs.model import MUST_AGREE, AttrsModel, parse_collection_text
from geosave_engine.geodata.attrs.palette import Palette

if TYPE_CHECKING:
    from collections.abc import Mapping


class Legend(AttrsModel):
    """Say what a label variable's pixel values mean, and how they colour.

    CF spells the listing two ways: `flag_values` enumerates the values a pixel
    takes and `flag_masks` names the bits a value packs flags into. `class_map`
    reads either back keyed by value, and a colour keys to a value it names.

    Args:
        flag_values: Pixel values, ascending and unique, one per class.
        flag_masks: Bit masks, for values packing several flags at once.
        flag_meanings: Space-separated class names, in `flag_values` order.
        color_map: `{pixel value: hex or RGB}`, keyed to the values listed.

    Raises:
        ValueError: The names and the codes differ in number, only one of the
            two is set, `flag_values` is not ascending and unique, or a colour
            keys to a value the listing does not name.

    Examples:
        >>> legend = Legend(class_map={0: "bg", 1: "palm"})
        >>> legend.flag_values, legend.flag_meanings
        ([0, 1], 'bg palm')
        >>> legend.class_map
        {0: 'bg', 1: 'palm'}
    """

    NAME: ClassVar[str] = "legend"

    flag_values: Annotated[
        list[int] | None, BeforeValidator(parse_collection_text), MUST_AGREE
    ] = None
    flag_masks: Annotated[
        list[int] | None, BeforeValidator(parse_collection_text), MUST_AGREE
    ] = None
    flag_meanings: Annotated[str | None, MUST_AGREE] = None
    color_map: Annotated[Palette | None, BeforeValidator(parse_collection_text)] = None

    if TYPE_CHECKING:
        # `class_map` spells the two CF fields, so Pydantic signs no keyword for it.
        def __init__(
            self,
            *,
            flag_values: list[int] | None = None,
            flag_masks: list[int] | None = None,
            flag_meanings: str | None = None,
            color_map: Palette | None = None,
            class_map: Mapping[int, str] | None = None,
        ) -> None: ...

    @property
    def class_map(self) -> dict[int, str] | None:
        """Return each pixel value mapped to the class it names.

        Returns:
            {
                <pixel value>: the class it names,
            }
            None where the listing enumerates no values, as a pure bitfield
            does.
        """
        if self.flag_values is None or self.flag_meanings is None:
            return None
        return dict(zip(self.flag_values, self.flag_meanings.split(), strict=True))

    @class_map.setter
    def class_map(self, class_map: Mapping[int, str]) -> None:
        """Rewrite the listing as a pixel-value-to-class mapping spells it.

        The codes and the names are one fact in CF's spelling, so both are
        written at once; writing either alone would name classes the other
        does not.

        Args:
            class_map: Pixel value mapped to the class it names. Class names
                carry no whitespace, which would split into stray
                `flag_meanings` tokens.

        Raises:
            ValueError: A class name is empty or carries whitespace, or the
                listing already masks a different number of classes.
        """
        spelled = _spelled(class_map)
        # Constructing checks the pair; assigning one at a time would not.
        listed = type(self)(
            flag_values=spelled.flag_values,
            flag_masks=self.flag_masks,
            flag_meanings=spelled.flag_meanings,
        )
        object.__setattr__(self, "flag_values", listed.flag_values)
        object.__setattr__(self, "flag_meanings", listed.flag_meanings)
        self.__pydantic_fields_set__.update({"flag_values", "flag_meanings"})

    @model_validator(mode="before")
    @classmethod
    def _spell_class_map(cls, data: Any) -> Any:
        """Accept a class map at construction, in the fields CF stores it in.

        Args:
            data: Field values, which may name `class_map` instead of the
                codes and names it spells out.

        Returns:
            The values with `class_map` replaced by the fields it fills.

        Raises:
            ValueError: A class name is empty or carries whitespace.
        """
        if not isinstance(data, dict) or "class_map" not in data:
            return data
        given = dict(data)
        spelled = _spelled(given.pop("class_map"))
        given["flag_values"] = spelled.flag_values
        given["flag_meanings"] = spelled.flag_meanings
        return given

    @model_validator(mode="after")
    def _check_the_listing_lines_up(self) -> Self:
        """Refuse a listing whose names and codes do not describe one another.

        Returns:
            This listing, unchanged.

        Raises:
            ValueError: Names accompany no codes or the reverse, a code list
                differs in length from the names, or `flag_values` is not
                ascending and unique.
        """
        codes = {"flag_values": self.flag_values, "flag_masks": self.flag_masks}
        listed = {name: found for name, found in codes.items() if found is not None}

        if self.flag_meanings is None:
            if listed:
                raise ValueError(
                    f"{sorted(listed)} list codes but flag_meanings names no "
                    f"classes for them; set it"
                )
            return self
        if not listed:
            raise ValueError(
                "flag_meanings names classes that no flag_values or flag_masks "
                "enumerate; set one of them"
            )

        names = self.flag_meanings.split()
        for field_name, found in listed.items():
            if len(found) != len(names):
                raise ValueError(
                    f"{field_name} lists {len(found)} codes but flag_meanings "
                    f"names {len(names)} classes"
                )
        if self.flag_values is not None and sorted(set(self.flag_values)) != (
            self.flag_values
        ):
            raise ValueError(
                f"flag_values {self.flag_values} are not ascending and unique"
            )

        # A colour for a value the listing does not name paints nothing.
        if self.color_map is not None and self.flag_values is not None:
            unlisted = sorted(set(self.color_map) - set(self.flag_values))
            if unlisted:
                raise ValueError(
                    f"colours {unlisted} key to values this listing does not "
                    f"name; it lists {self.flag_values}"
                )
        return self


class _Spelled(NamedTuple):
    """One class map as the two fields CF stores it in.

    Args:
        flag_values: The values ascending.
        flag_meanings: Their names in that order.
    """

    flag_values: list[int]
    flag_meanings: str


def _spelled(class_map: Mapping[int, str]) -> _Spelled:
    """Spell a pixel-value-to-class mapping in the fields CF stores it in.

    Args:
        class_map: Pixel value mapped to the class it names.

    Returns:
        The values ascending and their names in that order.

    Raises:
        ValueError: A class name is empty or carries whitespace, which would
            split into stray `flag_meanings` tokens.
    """
    split = sorted(name for name in class_map.values() if name.split() != [name])
    if split:
        raise ValueError(
            f"class names {split} are empty or carry whitespace; "
            f"flag_meanings tokens are single words, so use '_'"
        )
    values = sorted(class_map)
    return _Spelled(values, " ".join(class_map[value] for value in values))
