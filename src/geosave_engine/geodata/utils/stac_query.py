from typing import Any

class CQL2:
    """Static helpers for building CQL2-JSON filter expressions.

    All methods return plain dicts that can be nested freely and passed as
    the `filter` field of a `StacQuery`.  Use `and_` to combine conditions::

        CQL2.and_(
            CQL2.lt("eo:cloud_cover", 10),
            CQL2.eq("s2:nodata_pixel_percentage", 0),
        )
    
    example output:
        {
        "op": "and",
        "args": [
            {
                "op": "=",
                "args": [{ "property": "platform" }, "sentinel-2a"]
            },
            {
                "op": "<=",
                "args": [{ "property": "eo:cloud_cover" }, 5]
            },
            {
                "op": "in",
                "args": [
                    { "property": "s2:mgrs_tile" },
                    ["31UFU", "31UFV"]
                ]
            }
            ]
        }
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
    def neq(prop: str, value: Any) -> dict[str, Any]:
        return {"op": "<>", "args": [{"property": prop}, value]} # CQL2 uses <> for not equal

    @staticmethod
    def in_(prop: str, values: list[Any]) -> dict[str, Any]:
        return {"op": "in", "args": [{"property": prop}, values]}

    @staticmethod
    def between(prop: str, low: Any, high: Any) -> dict[str, Any]:
        """Check if property is between low and high (inclusive)."""
        return {"op": "between", "args": [{"property": prop}, [low, high]]}

    @staticmethod
    def is_null(prop: str) -> dict[str, Any]:
        """Check if property is null/missing."""
        return {"op": "is_null", "args": [{"property": prop}]}

    @staticmethod
    def like(prop: str, pattern: str) -> dict[str, Any]:
        return {"op": "like", "args": [{"property": prop}, pattern]}

    @staticmethod
    def and_(*exprs: dict[str, Any]) -> dict[str, Any]:
        return {"op": "and", "args": list(exprs)}

    @staticmethod
    def or_(*exprs: dict[str, Any]) -> dict[str, Any]:
        return {"op": "or", "args": list(exprs)}

    @staticmethod
    def not_(expr: dict[str, Any]) -> dict[str, Any]:
        return {"op": "not", "args": [expr]}

    # --- Spatial Operators ---

    @staticmethod
    def s_intersects(prop: str, geometry: dict[str, Any]) -> dict[str, Any]:
        """Spatial intersection between a property (usually 'geometry') and a GeoJSON dict."""
        return {"op": "s_intersects", "args": [{"property": prop}, geometry]}

    @staticmethod
    def s_contains(prop: str, geometry: dict[str, Any]) -> dict[str, Any]:
        return {"op": "s_contains", "args": [{"property": prop}, geometry]}

    # --- Temporal Operators ---

    @staticmethod
    def t_intersects(prop: str, interval: list[str]) -> dict[str, Any]:
        """Temporal intersection. Interval should be [start, end] ISO strings."""
        return {"op": "t_intersects", "args": [{"property": prop}, interval]}
