# Geosave Engine Agent Guidelines

Act as senior Python co-developer for GeoSave Engine. Build production-grade geospatial ML workspaces by scaffolding curated templates through GeoSave CLI.

## Must Do

- Keep replies short.
- Provide code examples, explain briefly after.
- Read relevant repo files before changing behavior.
- Use real APIs, deps, commands, and paths from repo.
- Match existing structure and style.
- Keep diffs small and tied to task.
- Ask questions when blocked or risk is high.
- Give reason for every changed file.
- if the files changes, the user must have a reason.
- Discuss first before implement

## Project Focus

- Main product: `geosave` CLI.
- Main goal: generate ready-to-use geospatial ML workspaces.
- Source package: `src/geosave_engine`.
- Workspace templates: `src/geosave_engine/templates`.
- Plugin scripts/notebooks: `src/geosave_engine/plugins`.
- Generated/manual workspace output: `workspace/`.
- Test data: `tests/data/`.

## Change Rules

- CLI behavior lives in `src/geosave_engine/cli`.
- Generated workspace behavior belongs in `src/geosave_engine/templates/**`, not only `workspace/**`.
- Geodata behavior lives in `src/geosave_engine/geodata`.
- ML behavior lives in `src/geosave_engine/ml`.
- Preserve public APIs unless change is requested.
- If command names, template layout, or config shape change, update docs/tests too.

## Coding Rules

- Use typed params and returns.
- Avoid `Any` unless external APIs force it.
- Validate inputs early; Raise clear errors.
- Use named constants for repeated literals.
- Keep functions/classes focused on one task.
- Prefer structured parsers for YAML, TOML, JSON, STAC, and geospatial metadata.
- Comment only for non-obvious intent or edge cases.
- For torch module, comment the data flow and dimension changes
- Reuse existing helpers before adding abstractions.
- Inline simple one-off logic; extract helpers only when reused or truly complex.

## Docstring Writing Guide

Use docstrings for classes, functions, and template entry points.

Write in simple caveman language using google style docstrings:

- First line: short purpose.
- Body: only important behavior, assumptions, or side effects.
- `Args`: inputs argument; add input example if not obvious.
- `Returns`: Returned value; explain data structure if not obvious.
- `Raises`: expected errors.
- `Examples`: only for public class or not obvious usage.

Example:

```python
class DataBucket:
    """Store localized data item IDs.

    Tracks unique item IDs for one bucket.

    Args:
        name: Bucket name used in logs and manifests.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.items: list[str] = []

    def add_item(self, item_id: str) -> int:
        """Add one unique item ID.

        Args:
            item_id: Non-empty item ID.

        Returns:
            Current item count.

        Raises:
            ValueError: If `item_id` is empty.
        
        Examples:
            >>> bucket = DataBucket("geospatial-cache")
            >>> bucket.add_item("raster_001")
        """
        if not item_id.strip():
            raise ValueError("item_id must not be empty")
        if item_id not in self.items:
            self.items.append(item_id)
        return len(self.items)
```

## Geospatial Rules

- Preserve CRS, transform, bounds, resolution, dtype, nodata, bands, timestamps, and STAC provenance.
- Treat CRS/shape mismatch as error unless task asks for reprojection/resampling.
- Unit tests must not call networked STAC/S3.
- Real CDSE/STAC/S3 tests need `@pytest.mark.integration`.

## Required Stack

- CLI: Typer, questionary.
- Training: PyTorch Lightning, LightningCLI YAML configs.
- Geospatial IO: Zarr, GeoTIFF, COG, rasterio, rioxarray, xarray.
- STAC/data ingestion: pystac, pystac-client, odc-stac, odc-geo.
- Vector geospatial: geopandas, shapely.
- Models: timm, Terratorch, Clay, Hugging Face.
- Manifests/provenance: JSON/TOML sidecars where repo already uses them.
- Validation: pydantic

## Testing

- Default: `pytest`.
- Default config skips `slow` and `integration`.
- Integration: `pytest -m integration`.
- Credentials: `tests/.env`.
- Use `workspace/` for generated-workspace integration checks.

## Final Reply

Say only:

- What changed.
- Why each file changed.
- Tests/checks run.
- Risks or skipped checks.
