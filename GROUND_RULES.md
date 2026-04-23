# Ground Rules

These rules are **binding**, not aspirational. Read this before the file tree.
New code must follow them; old code gets brought into compliance when it's touched.

The goal is a library that is **small, strict, and predictable** — one where a reader
can trust that a function either returns a valid result or raises, and where the
shape of data never has to be guessed from context.

---

## Layout

The package is organized by the *kind of dependency* each subpackage pulls in, not
by feature. This keeps the dependency graph one-directional and lets us reason
about what gets loaded when something is imported.

1. **`geodata/`** — STAC + raster I/O. May import `rasterio`, `rioxarray`, `odc.*`,
   `pystac*`, `pyproj`, `shapely`. **Must not** import torch, lightning, or anything
   from `ml/`.
2. **`ml/`** — torch / lightning / torchmetrics / segmentation-models-pytorch.
   May import from `utils/` and `core/`. **Must not** import from `geodata/` or `cli/`.
3. **`utils/`** — pure functions over stdlib and scientific-Python types
   (numpy, shapely, pyproj). No business logic. No imports from sibling
   subpackages (`utils/` is a leaf).
4. **`cli/`** — the top of the stack. May import anything, but uses AST-based
   discovery and lazy imports so that `import geosave_engine.cli` stays fast.
5. Base classes lives in core module for each sub module liek ml/core and geodata/core

If a change would break one of these directions (e.g. `geodata/` needing torch),
that is a signal to redesign, not to bend the rule.

---

## Coding

### 1. No silent fallbacks

Functions that can't produce a valid result **raise**. `return None` is reserved
for cases where absence is a semantically valid answer (e.g. `dict.get("optional_key")`),
not for "something went wrong".

Banned patterns:

```python
# NO — masks missing data as zero
sun_azimuth = item.properties.get("view:sun_azimuth", 0.0)

# NO — masks missing attribute silently
extra = getattr(asset, "extra_fields", {}) or {}

# NO — swallows all errors, caller can't distinguish failure modes
try:
    result = load(...)
except Exception:
    return None
```

Replacement patterns:

```python
# Raise when the key is required
if "view:sun_azimuth" not in item.properties:
    raise ValueError(f"STAC item {item.id!r} missing required field 'view:sun_azimuth'")
sun_azimuth = float(item.properties["view:sun_azimuth"])

# Or: catch narrowly and classify
try:
    result = load(...)
except RasterioIOError as exc:
    raise IngestionFailed(f"could not load {path}") from exc
```

`None` is a fine return **only** when the function is semantically
"look up X, tell me if it's there". A loader is not such a function.

### 2. No union types at public API boundaries

A public function signature should name **one** type per parameter. Unions let
bugs hide: every caller needs to know whether they hit the `str` branch or the
`Path` branch.

```python
# NO
def load(paths: str | PathLike[str] | list[str | PathLike[str]]) -> ...

# YES
def load(paths: Sequence[Path]) -> ...
```

Caller converts. `T | None` is allowed only when `None` carries a distinct
meaning (e.g. "not yet configured"), not as a way to say "optional".

### 3. No `Any`

Use `Protocol` for structural typing, `TypeVar` for generics, or concrete types.
`Any` is acceptable only where the value crosses a boundary into an external
library whose types we don't control (e.g. pystac-client's `query=` dict).
Document the acceptable shape in that case.

### 4. No `# type: ignore` without reason

Every `# type: ignore` in library code must be paired with a `# reason: …`
comment explaining what is being suppressed and why the checker is wrong.
Goal: zero ignores in `src/geosave_engine/`.

### 5. Dataclasses for parameter bundles

If three or more fields travel together through a function call chain, they are
a dataclass. `@dataclass(frozen=True)` by default. Mutable only when there's a
specific reason.

```python
# NO — dict leaks across four call sites
def load(..., rio_kwargs: dict | None = None): ...

# YES
@dataclass(frozen=True)
class GeoTiffOptions:
    compress: str = "lzw"
    predictor: int = 2
    tiled: bool = True

def load(..., options: GeoTiffOptions = GeoTiffOptions()): ...
```

### 6. Magic strings live in `constants.py`

STAC collection IDs, STAC URLs, CQL2 filter keys, GDAL environment keys, and
other fixed external identifiers belong in `geodata/constants.py` (or a
subpackage equivalent). One module owns them; everyone else imports.

### 7. `logging`, not `print`

Library code uses the `logging` module. `print()` is allowed only in:

- Scripts under `scripts/` or `wawa/scripts/`
- CLI entry points (`cli/main.py` command handlers writing to the terminal)

Inside a `tqdm` loop, use `tqdm.write(...)` to avoid corrupting the progress bar.

### 8. Docstrings describe contracts

Docstrings say what the function **promises** and what it **demands**:

- Inputs and their constraints (what counts as valid)
- What it returns
- What it raises and when
- Any invariant it maintains

They do not line-by-line narrate the body. The code does that.

```python
def extract_scale_offset_from_item(item: pystac.Item) -> tuple[float, float]:
    """Return (scale, offset) from the first asset that has radiometry metadata.

    Raises:
        RadiometryMetadataMissing: no asset carries a non-zero ``raster:scale``.
    """
```

### 9. Naming

- No abbreviations except ones universally understood in geospatial work:
  `crs`, `bbox`, `utm`, `epsg`, `lat`, `lon`, `dn` (digital number), `toa`, `sr`.
- **Banned** abbreviations: `svc`, `da`, `meta`, `ds`, `idx` (in non-trivial
  scope), `cfg` in public signatures.
- Preferred: `ingestor`, `data_array`, `metadata`, `dataset`, `index`, `config`.
- Classes name the thing they are (`Sentinel2Ingestor`, `CdseClient`). Functions
  name the action (`extract_scale_offset_from_item`, `read_tiff_metadata`).
- Private helpers are prefixed `_`. Module-private constants are `UPPER_SNAKE`
  without the underscore unless they are genuinely an implementation detail
  that must not leak.

---

## Process

- **Each logical change is one commit.** A dataclass introduction and the
  callers that use it go together; a rename pass is its own commit.
- **Verification runs after every phase** — see the refactor plan's verification
  section for the standard checks (imports resolve, ruff clean, `print_config`
  on the wawa training config succeeds).
- Changes that affect the public API of `geodata/`, `ml/`, or `core/` must
  update `CHANGELOG.md` (create if missing).

---

## When a rule needs to be broken

Rules exist to make the common case simple. If you hit a genuine exception:

1. Write the comment explaining *why* — future-you (or a reviewer) needs to
   know this was a conscious decision, not an oversight.
2. Keep the exception narrow. A `# reason:` on a single ignore is fine; a
   whole file that needs escape hatches is a signal that the design is
   wrong upstream.
3. If the same exception comes up three times, the rule is wrong — raise it
   for discussion rather than accumulating exceptions.
