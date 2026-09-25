from __future__ import annotations

from enum import Enum, auto


class Unset(Enum):
    """Sentinel type for a keyword argument that was not passed.

    Distinct from an explicit None, and single-member so that `is not UNSET`
    narrows a `X | None | Unset` union down to `X | None`.
    """

    TOKEN = auto()


UNSET = Unset.TOKEN
