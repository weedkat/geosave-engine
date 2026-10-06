# Name Format Strings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> Deferred on 2026-10-06 with its spec; do not execute until the key list is settled.

**Goal:** Render a raster's name from a format string over keys read from its anchor.

**Architecture:** `GeoAnchor.fields` exposes the values `stem` already computes
as named keys; `GeoAnchor.format` hands them to `str.format`. Nothing parses a
name back.

**Tech Stack:** Python `str.format`, odc-geo, existing `geodata.utils` tokens.

**Spec:** `docs/superpowers/specs/2026-10-06-name-format-design.md`. The key
table there is binding; do not add keys.

## Global Constraints

- The checkout is dirty on `main` and shared. Do not commit, stash, checkout or restore.
- No new dependencies, no parsing of names, no tiling or place keys.
- No private helper beyond what this plan shows. If a step needs more code than
  shown, stop and report.
- Google-style docstrings checked with `python scripts/check_docstrings.py`.

## Review Focus

1. A geographic CRS (degrees): `res` and `extent` render in degrees, and `x`/`y`
   equal `lon`/`lat` (Task 1).
2. A CRS with no EPSG code: `epsg` is absent and `{epsg}` raises `KeyError` (Task 1).
3. A caller keyword repeating a key raises `TypeError` instead of overriding (Task 1).

---

### Task 1: `GeoAnchor.fields` and `GeoAnchor.format`

**Files:**
- Modify: `src/geosave_engine/geodata/core/anchor.py`
- Test: the file that tests `GeoAnchor.stem` today (`grep -rln "anchor.stem\|\.stem ==" tests/geodata`); add the tests there

**Interfaces:**
- Produces: `GeoAnchor.fields -> dict[str, object]`,
  `GeoAnchor.format(template: str, **extra: object) -> str`. `stem` is unchanged
  in output.

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime

import pytest
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import GeoAnchor

from tests.geodata.conftest import build_raster


def test_fields_name_what_the_stem_is_built_from() -> None:
    anchor = build_raster(times=2).gs.anchor

    fields = anchor.fields

    assert fields["start"] == datetime(2025, 6, 1)
    assert fields["end"] == datetime(2025, 6, 2, 23, 59, 59, 999999)
    assert (fields["epsg"], fields["width"], fields["height"]) == (32749, 2, 2)
    assert fields["lon"] == pytest.approx(111.0001, abs=1e-4)
    assert fields["lat"] == pytest.approx(-9.0465, abs=1e-4)
    assert fields["x"] == pytest.approx(500_010.0)
    assert fields["y"] == pytest.approx(9_000_010.0)
    assert anchor.stem == "_".join(
        fields[key] for key in ("centroid", "extent", "dates", "res")
    )


def test_timeless_fields_carry_no_time_keys() -> None:
    anchor = build_raster().gs.anchor

    assert not {"start", "end", "dates"} & anchor.fields.keys()
    assert anchor.stem == "_".join(
        anchor.fields[key] for key in ("centroid", "extent", "res")
    )
    with pytest.raises(KeyError, match="start"):
        anchor.format("{start:%Y}")


def test_format_renders_keys_specs_and_caller_keywords() -> None:
    anchor = build_raster(times=2).gs.anchor

    assert anchor.format("{start:%Y}-{start:%m}_{lon:.2f}_{lat:.2f}") == (
        "2025-06_111.00_-9.05"
    )
    assert anchor.format("{site}_{end:%Y%m%d}_{res}", site="forest") == (
        "forest_20250602_10m"
    )


def test_format_refuses_unknown_and_repeated_keys() -> None:
    anchor = build_raster(times=2).gs.anchor

    with pytest.raises(KeyError, match="mgrs"):
        anchor.format("{mgrs}")
    with pytest.raises(TypeError):
        anchor.format("{res}", res="20m")


def test_a_geographic_grid_renders_degrees_and_no_missing_epsg() -> None:
    anchor = build_raster(crs="EPSG:4326").gs.anchor

    assert anchor.fields["x"] == pytest.approx(anchor.fields["lon"])
    assert anchor.fields["res"].endswith("deg")

    laea = GeoBox.from_bbox(
        (0, 0, 40, 40),
        crs="+proj=laea +lat_0=12.34 +lon_0=56.78 +datum=WGS84 +units=m",
        resolution=10,
    )
    assert "epsg" not in GeoAnchor(laea).fields


def test_a_rendered_name_becomes_the_folder_and_item_prefix(tmp_path) -> None:
    from geosave_engine.geodata import GeoVector

    raster = build_raster(times=1)
    name = raster.gs.anchor.format("{site}_{start:%Y%m}", site="forest")

    paths = raster.gs.to_cog(tmp_path / name)

    assert paths[0].parent.name == "forest_202506"
    assert GeoVector.from_rasters(paths)["id"].tolist() == [
        "forest_202506_20250601T000000"
    ]
```

- [ ] **Step 2: Run** `uv run pytest <that test file> -k "fields or format or rendered_name" -q`
  Expected: FAIL with `AttributeError: 'GeoAnchor' object has no attribute 'fields'`.

- [ ] **Step 3: Implement** in `core/anchor.py`. Replace the body of `stem` and
  add the two members beside it:

```python
    @property
    def fields(self) -> dict[str, object]:
        """Return the keys a name format string can use.

        Returns:
            {
                "start" | "end": first and last covered instant, when dated,
                "dates": compact token for that period, when dated,
                "lon" | "lat": grid centroid in WGS84 degrees,
                "centroid": centroid token such as `111.0001E_9.0465S`,
                "x" | "y": grid centroid in the grid's CRS,
                "epsg": EPSG code of the CRS, when it has one,
                "res": pixel size token such as `10m`,
                "extent": ground extent token such as `5kmx5km`,
                "width" | "height": grid size in pixels,
            }

        Examples:
            >>> anchor.fields["res"]
            '10m'
        """
        longitude, latitude = self.geographic_centroid
        unit = self.crs.units[0]
        bounds = self.geobox.boundingbox
        x_size, y_size = self.resolution.map(abs).xy
        resolution = format_ground_size(x_size, unit)
        if not math.isclose(x_size, y_size):
            resolution += f"x{format_ground_size(y_size, unit)}"
        x, y = self.geobox.extent.centroid.coords[0]
        fields: dict[str, object] = {
            "lon": longitude,
            "lat": latitude,
            "centroid": (
                f"{abs(longitude):.4f}{'E' if longitude >= 0 else 'W'}_"
                f"{abs(latitude):.4f}{'N' if latitude >= 0 else 'S'}"
            ),
            "x": float(x),
            "y": float(y),
            "res": resolution,
            "extent": (
                f"{format_ground_size(bounds.span_x, unit)}x"
                f"{format_ground_size(bounds.span_y, unit)}"
            ),
            "width": self.geobox.width,
            "height": self.geobox.height,
        }
        if self.crs.epsg is not None:
            fields["epsg"] = self.crs.epsg
        if self.timespan is not None:
            start, end = (naive_utc(edge) for edge in self.timespan)
            fields.update(start=start, end=end, dates=format_stem_dates((start, end)))
        return fields

    def format(self, template: str, **extra: object) -> str:
        """Render a name from a format string over this anchor's fields.

        Args:
            template: `str.format` template naming keys of `fields`, each with
                an optional format spec, such as `"{start:%Y%m}_{res}"`.
            **extra: Caller keys the template may also name.

        Returns:
            The rendered name.

        Raises:
            KeyError: The template names a key neither `fields` nor `extra` has.
            TypeError: `extra` repeats a key of `fields`.

        Examples:
            >>> anchor.format("{site}_{start:%Y%m}", site="forest")
            'forest_202506'
        """
        return template.format(**self.fields, **extra)

    @property
    def stem(self) -> str:
        """<keep the existing docstring unchanged>"""
        fields = self.fields
        return "_".join(
            str(fields[key])
            for key in ("centroid", "extent", "dates", "res")
            if key in fields
        )
```

Keep `stem`'s existing docstring text; only its body changes.

- [ ] **Step 4: Run** `uv run pytest tests/geodata -q && uv run ruff check src tests && python scripts/check_docstrings.py src/geosave_engine/geodata/core/anchor.py`
  Expected: PASS and clean. Every existing `stem` assertion must still pass
  unchanged; if one changes, stop and report.

- [ ] **Step 5: Document.** Add the format-string example from the spec's first
  code block to `docs/guides/architecture.md` beside the catalog loop.
