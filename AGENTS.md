# GeoSave Engine Agent Guidelines

## Role

Act as a senior Python co-developer for GeoSave Engine. Build production-grade geospatial ML tooling and workspace templates while keeping the library small, explicit, and maintainable.

The `geosave` CLI is the product entry point. The repository also owns the geodata and ML implementations used by generated workspaces.

## Working agreement

- Keep replies concise.
- Inspect relevant source, configuration, and current call sites before changing behavior.
- For architecture, public behavior, or breaking changes, discuss the design and tradeoffs before implementation.
- Implement scoped fixes directly once intent is clear. Ask only when a missing choice materially changes behavior or risk.
- Explain a better design when one is available.
- Keep diffs focused. Preserve unrelated work in a dirty worktree.
- Give a reason for every changed file.
- Use real repository APIs, dependencies, commands, and paths. Verify documentation claims against source because docs may lag during redesign.
- Preserve public interfaces unless the user requests or approves a break.

## Source ownership

- CLI behavior: `src/geosave_engine/cli`.
- Generated workspace behavior: `src/geosave_engine/templates/**`.
- Geospatial behavior: `src/geosave_engine/geodata`.
- ML behavior: `src/geosave_engine/ml`.
- Test fixtures: `tests/data`.
- `workspace/` is a generated consumer/example, not library source. Fix template behavior in `src/geosave_engine/templates/**`.

## Design and coding

- Apply SOLID, DRY, YAGNI, and KISS. Prefer one strict contract over compatibility glue.
- Keep modules focused and interfaces smaller than their implementations.
- Separate format adapters from domain behavior. Do not leak GDAL, CF, STAC, or ML-specific normalization into Spatial core.
- Validate inputs early and raise actionable errors that identify the mismatch and corrective operation.
- Use typed parameters and returns. Avoid `Any` unless an external library forces it.
- Prefer explicit transformations over silent inference, repair, reprojection, resampling, sorting, broadcasting, or dtype promotion.
- Reuse an existing helper when it owns the same invariant. Inline simple one-off logic; extract code only when it improves locality or is reused.
- Use structured parsers for YAML, TOML, JSON, STAC, and geospatial metadata.
- Prefer readable, predictable code over clever or line-saving implementations.
- Use dependencies declared in `pyproject.toml`; discuss new dependencies or stack substitutions first.

## Spatial invariants

- Canonical raster data is an `xr.DataArray` with dimensions `(band, y, x)` or `(time, band, y, x)`.
- `band` is mandatory and explicitly named. Raw NumPy input requires band names.
- Preserve CRS, transform, bounds, resolution, dtype, nodata, bands, timestamps, vector data, and STAC provenance.
- Treat grid, CRS, shape, band, time, dtype, and nodata mismatches as errors unless the caller explicitly requests the required transformation.
- Composition is strict. Callers reproject, resample, select, rename, or cast before composing.
- Keep lazy arrays lazy unless the interface explicitly materializes pixels.
- Unit tests must not access networked STAC or S3. Mark real CDSE/STAC/S3 checks with `@pytest.mark.integration`.

## Documentation and comments

- Use concise Google-style docstrings for public classes, functions, and template entry points.
- State purpose, inputs, outputs, constraints, and expected errors. Do not narrate implementation flow, history, rationale, sibling comparisons, or rejected designs.
- Document constructor fields in a class docstring. Do not repeat the class description in every method.
- Every public callable documents caller-supplied parameters under `Args` and every non-`None` result under `Returns`. Omit sections that do not apply; never add empty `Args` or a fake `Returns: None`.
- Add `Examples` when construction, configuration, or a transformation is not obvious from the signature. Use real public APIs and omit unrelated setup.
- A simple property still documents its value under `Returns`.
- Put flow in code structure. Use a short standalone comment only for non-obvious chunks or edge cases.
- Reserve trailing comments for mechanical notes such as tensor shapes.
- For Torch modules, annotate important tensor dimension changes.
- Run `python scripts/check_docstrings.py <file_or_dir>` after changing docstrings or comments.

Use these examples as style boundaries. Include only the sections that help that API.

A public class describes the object, constructor inputs, construction constraints, and a minimal usage path:

```python
@dataclass(frozen=True, eq=False)
class GeoVector:
    """Store vector geometries and properties in one CRS.

    Args:
        gdf: Non-empty GeoDataFrame with a CRS and valid geometries.

    Raises:
        ValueError: If the CRS is missing or any geometry is null, empty, or invalid.

    Examples:
        >>> vector = GeoVector.open("plantations.geojson")
        >>> vector.gdf[["geometry", "crop"]]
    """
```

A public function or method documents its contract and demonstrates non-obvious usage:

```python
def rename_bands(self, mapping: dict[str, str]) -> Self:
    """Rename bands without changing their order or pixels.

    Args:
        mapping: Existing band names mapped to replacement names.

    Returns:
        New raster with renamed band coordinates.

    Raises:
        KeyError: If a source band is absent.
        ValueError: If a replacement is empty or creates a duplicate.

    Examples:
        >>> raster.bands
        ("B04", "B08")
        >>> renamed = raster.rename_bands({"B04": "red", "B08": "nir"})
        >>> renamed.bands
        ("red", "nir")
    """
```

Keep obvious accessors short while documenting their result:

```python
@property
def band_count(self) -> int:
    """Return the number of raster bands.

    Returns:
        Number of bands.
    """
```

Avoid docstrings that narrate implementation or compare designs:

```python
def rename_bands(...):
    """First check duplicates, then replace names, unlike the old dataset helper."""
```

Comments describe the next non-obvious code chunk. Shape notes may remain trailing:

```python
# Pool spatial features for the classification head.
features = self.pool(features)  # (batch, channels, 1, 1)
features = features.flatten(1)  # (batch, channels)
```

Do not restate syntax, narrate every step, or preserve design history in comments:

```python
# Call flatten to flatten the tensor from four dimensions to two dimensions.
features = features.flatten(1)
```

Documentation may remain skeletal during an explicitly agreed redesign. Once behavior is settled, update the affected docs and tests before building dependent interfaces.

## Verification

- Match checks to risk: targeted tests for local behavior, round-trip tests for persistence, and smoke tests during design work.
- Default suite: `pytest` (skips `slow` and `integration` through project configuration).
- Integration suite: `pytest -m integration`; credentials live in `tests/.env`.
- Use `workspace/` only for generated-workspace integration checks.
- Run relevant lint, type, compile, formatting, and docstring checks before handoff.
- Report stale or intentionally skipped checks; do not claim success from unrelated passing tests.

## Final response

Report only:

- What changed and why each file changed.
- Tests and checks run.
- Risks, breaking changes, and skipped checks.
