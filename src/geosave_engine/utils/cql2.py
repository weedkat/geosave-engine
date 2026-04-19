from __future__ import annotations

from typing import Any


class CQL2:
    """Static helpers for building CQL2-JSON filter expressions.

    All methods return plain dicts that can be nested freely and passed as
    the `filter` field of a `StacQuery`.  Use `and_` to combine conditions::

        CQL2.and_(
            CQL2.lt("eo:cloud_cover", 10),
            CQL2.eq("s2:nodata_pixel_percentage", 0),
        )
    """

    @staticmethod
    def lte(prop: str, value: Any) -> dict[str, Any]:
        return {"op": "<=", "args": [{"property": prop}, value]}

    @staticmethod
    def gte(prop: str, value: Any) -> dict[str, Any]:
        return {"op": ">=", "args": [{"property": prop}, value]}

    @staticmethod
    def lt(prop: str, value: Any) -> dict[str, Any]:
        return {"op": "<", "args": [{"property": prop}, value]}

    @staticmethod
    def gt(prop: str, value: Any) -> dict[str, Any]:
        return {"op": ">", "args": [{"property": prop}, value]}

    @staticmethod
    def eq(prop: str, value: Any) -> dict[str, Any]:
        return {"op": "=", "args": [{"property": prop}, value]}

    @staticmethod
    def and_(*exprs: dict[str, Any]) -> dict[str, Any]:
        return {"op": "and", "args": list(exprs)}

    @staticmethod
    def or_(*exprs: dict[str, Any]) -> dict[str, Any]:
        return {"op": "or", "args": list(exprs)}

    @staticmethod
    def not_(expr: dict[str, Any]) -> dict[str, Any]:
        return {"op": "not", "args": [expr]}
