# Geodata simplification sweep

Implementation status: the subsequent [data-flow plan](../superpowers/plans/2026-09-23-geodata-dataflow.md) records the changes, review refinements, and tests. Findings below describe the pre-refactor snapshot.

Reviewed the current working tree on 2026-09-23 in this order: attrs, core,
stac, transform, features, pipeline. This is a review, not an implementation
plan. Existing source changes were preserved. HEAD at review time was
`0f96a4c846b6a49382891e7850292cfdc649a463`; findings concern the working tree,
including uncommitted files.

The most useful simplifications concentrate existing invariants: attrs writes,
metadata conversion, STAC decoding, and preservation of native xarray objects.
Keep the registered attrs models, native xarray accessors, explicit transforms,
and geodata-owned tiling. Their deletion would spread real complexity into
callers. There is no evidence here for a replacement framework.

## 1. Attrs

### 1. Prevalidate every metadata patch before writing — small

**Files:** [attrs/xarray.py](../../src/geosave_engine/geodata/attrs/xarray.py),
especially `_rebase_namespace` and `_rebase_models`.

**Evidence:** This raises for `missing` after already changing `red`:

```python
attrs.rebase(
    ds, Nodata(fill_value=0), target=["red", "missing"], inplace=True
)
```

**Simplification:** Keep the `rebase` Interface and concentrate target resolution
and patch preparation before mutation. Header application already resolves its
targets first; the other two branches duplicate a weaker Implementation.
This gives the Module consistent error behavior and improves Locality without
adding a new Interface. The Leverage is that callers need no recovery copy for
an invalid target. Extend the existing header failure test to model and namespace
patches, including a bad second target and non-mutation on failure.

### 2. Make shared-key agreement a namespace invariant — small/medium

**Files:** [attrs/namespace.py](../../src/geosave_engine/geodata/attrs/namespace.py)
and [attrs/model.py](../../src/geosave_engine/geodata/attrs/model.py).

**Evidence:** Parsing `{"units": "metre"}` creates both `CFVariable` and
`CFCoordinate`. Setting `namespace.get(CFVariable).units = "kilometre"` succeeds,
but `namespace.to_attrs()["units"]` remains `"metre"`. Serialization silently
chooses the later model. Registration checks matching field types, not the
agreement of subsequently edited values.

**Simplification:** Have the namespace Module detect contradictory emitted
values at its serialization Seam, including explicit clears. Keep the useful
registered models. This concentrates conflict knowledge in one Implementation
and gives callers predictable Leverage from typed edits. Tests should use the
public namespace Interface with mutation, conflicting direct construction, and
different model insertion orders. Introducing rejection is a behavior change
for callers currently relying on last-write wins.

## 2. Core

### 3. Schedule band statistics together — small

**Files:** [core/array.py](../../src/geosave_engine/geodata/core/array.py),
`GeoArray.statistics`, lines 496–507.

**Evidence:** A 2×2 delayed source block executes **five times** for one call:
count, minimum, maximum, mean, and standard deviation each trigger computation.
Putting the scalar reductions into one native xarray Dataset and computing it
once executes the same source block **once**, with equal numerical results.

**Simplification:** Keep `statistics()` and `BandSummary` as the Module's
Interface; jointly schedule the reductions inside its Implementation. This
improves I/O Locality and gives callers the Leverage promised by one summary
operation. Do not materialize the entire raster just to share reductions.
Extend the eager numerical tests with a delayed read counter and all-nodata
coverage.

### 4. Preserve root metadata explicitly during band stacking — medium

**Files:** [core/raster.py](../../src/geosave_engine/geodata/core/raster.py),
`to_array`; [core/array.py](../../src/geosave_engine/geodata/core/array.py),
`to_raster`; [attrs/models/band.py](../../src/geosave_engine/geodata/attrs/models/band.py).

**Evidence:** A Dataset with root `units="root-unit"` and a band with
`units="band-unit"` returns root attrs `{}` after
`ds.gs.to_array().gs.to_raster()`. Stacking overwrites overlapping root keys;
restoration subtracts all keys appearing in restored band attrs.

**Simplification:** Concentrate reversible metadata ownership in the existing
`BandVariables` Module instead of inferring it independently on both sides of
the conversion Seam. Retain original root metadata as well as per-band metadata.
That improves Locality and the Leverage of the existing round-trip Interface.
Test overlapping root/band keys, selected bands, laziness, and persisted
round trips. The representation and treatment of later array-attr edits need
a design decision before implementation.

## 3. STAC

### 5. Preserve the supplied client's transport — small

**Files:** [stac/client.py](../../src/geosave_engine/geodata/stac/client.py),
lines 38–47 and catalog factory methods.

**Evidence:** `StacClient(client)` replaces the supplied client's `_stac_io`.
An offline smoke with a custom header and seven-second timeout loses both.
The installed pystac-client uses this transport for subsequent searches.

**Simplification:** Remove transport replacement from the wrapping constructor.
Configure retry behavior when the catalog factories create their own native
client. The supplied transport remains the Adapter at that Seam. This reduces
the Module's hidden Interface requirements and improves configuration Locality;
callers gain Leverage from their already-configured client. Test transport
identity, headers, timeout, and request modifier without external requests.

### 6. Stamp metadata from the effective band selection and decoding — high priority, medium

**Files:** [stac/source.py](../../src/geosave_engine/geodata/stac/source.py),
`load`; [stac/stamp.py](../../src/geosave_engine/geodata/stac/stamp.py),
`read_header` and `shared_asset_fields`.

**Evidence:** With real local GeoTIFF assets:

```text
config: dtype=float32, nodata=-9999
loaded missing pixel: -9999
returned attrs: _FillValue=0, nodata=0

config alias: red -> B04
loaded variable: red
returned variable attrs: {}  # packing and units were lost
```

The loader consumes configuration, but stamping independently looks up raw
assets by output variable name and replaces the loaded attrs. This creates two
interpretations of the same pixels.

**Simplification:** Make the loading Module own one interpretation of effective
band identity and decoding. Reuse ODC's native band resolution, account for
overrides, and keep source provenance distinct from output decoding attrs.
Do not add another asset resolver. This deepens the existing loading Interface:
one call produces mutually consistent pixels and metadata. Locality improves
because alias and override fixes apply to loading and stamping together.
Test aliases, configured nodata, multi-band assets, conflicting source metadata,
and overrides through the full local-file loading path, then mask/unpack the
result. These gaps are not exercised by the nine geodata STAC tests.

## 4. Transform

### 7. Normalize nodata aliases at the xarray codec Seam — high priority, small

**Files:** [transform/nodata.py](../../src/geosave_engine/geodata/transform/nodata.py),
`_to_nan_array`, lines 75–83.

**Evidence:** `to_nan` on pixels `[0, 10]` with only `attrs={"nodata": 0}`
returns `[0, 10]` and empty attrs. The typed model recognizes `nodata`, but
`CFMaskCoder` receives the original attrs without `_FillValue`; the marker is
then removed anyway. Normalizing through the existing attrs model first yields
`[NaN, 10]` in the smoke check.

**Simplification:** Use the already-parsed `Nodata` model to supply the codec's
canonical spelling. Keep masking and unpacking explicit. This concentrates
alias knowledge in the existing Module and gives its Interface consistent
Leverage for native ODC and CF inputs. Test both spellings separately and
together, NaN fill, packed inputs, and Dask laziness.

### 8. Preserve native structure when replacing pixels — medium

**Files:** [transform/nodata.py](../../src/geosave_engine/geodata/transform/nodata.py),
[transform/packing.py](../../src/geosave_engine/geodata/transform/packing.py),
and DataTree reconstruction in
[transform/concat.py](../../src/geosave_engine/geodata/transform/concat.py),
[transform/warp.py](../../src/geosave_engine/geodata/transform/warp.py), and
[transform/time.py](../../src/geosave_engine/geodata/transform/time.py).

**Evidence:** Dataset `to_nan` and `unpack` lose coordinates on dimensions not
used by a data variable, even when no pixels require conversion. Masking a stack
with root `title="keep-root"` returns a stack with no root attrs. Several
transforms rebuild containers independently from their children.

**Simplification:** Preserve the existing native container where its structure
is unchanged; where a transform changes the grid or time axis, explicitly
retain unaffected metadata. Concentrate that preservation rule within each
transform Module instead of making callers restore it. This improves Locality
and the Leverage of the existing transform Interface. Native `Dataset.map`
alone is insufficient: a smoke check shows it also drops disconnected coords.
Test no-op transforms, auxiliary coordinates and their attrs, root metadata,
changed spatial metadata, and lazy arrays.

## 5. Features

### 9. Retain the reference coordinates in halo-derived fields — small/medium

**Files:** [utils/dask.py](../../src/geosave_engine/geodata/utils/dask.py),
lines 79 and 104; callers in
[features/cloud_mask.py](../../src/geosave_engine/geodata/features/cloud_mask.py)
and [features/shadow_mask.py](../../src/geosave_engine/geodata/features/shadow_mask.py).

**Evidence:** A georeferenced shadow-mask input carrying `sun_azimuth(time)`
loses that coordinate and its time-coordinate attrs. The halo helper rebuilds
the result from a geobox and leading dimension labels instead of retaining the
reference's full coordinate structure.

**Simplification:** Let the halo Module replace values while preserving the
reference dimensions and coordinates. Define derived-value attrs separately
so reflectance packing is not inherited by a boolean mask. This improves
Locality across cloud and shadow functions, and gives their existing Interface
the Leverage of preserving native acquisition coordinates. Test eager/lazy
parity, chunk edges, coordinate attrs, CRS, and auxiliary time coordinates.

### 10. Make feature alignment explicit — small/medium

**Files:** [features/spectral_indices.py](../../src/geosave_engine/geodata/features/spectral_indices.py)
and input checks in [utils/dask.py](../../src/geosave_engine/geodata/utils/dask.py).

**Evidence:** NDVI inputs with x labels `[0, 1]` and `[1, 2]` silently produce
only x label `[1]`. Arithmetic intersects indexes before `_ratio` can inspect
the original inputs. The halo path already requests exact index alignment,
but that alone does not establish CRS agreement.

**Simplification:** Keep the small named formula Modules; establish input
agreement before arithmetic using native exact alignment and applicable grid
checks. Do not replace the formulas with a configurable expression registry.
The Interface then consistently requires aligned bands, concentrating Locality
and giving callers Leverage through explicit failure instead of silent cropping.
Tests should cover shifted indexes, incompatible georeferencing, dimension
broadcasting, and Dask inputs, alongside the existing numerical examples.

### 11. Remove feature-owned automatic unpacking — small behavior change

**Files:** [features/cloud_mask.py](../../src/geosave_engine/geodata/features/cloud_mask.py),
`compute_s2c_mask`, lines 94–96.

**Evidence:** This function promises automatic unpacking, but an ordinary packed
band with nodata fails immediately: the transform correctly requires `to_nan`
before `unpack`. Other reflectance feature functions expect callers to prepare
physical values explicitly.

**Simplification:** Require explicitly prepared reflectance consistently and
remove the implicit unpacking from this feature Module. Radiometric preparation
stays at the transform Seam. The deletion reduces Interface surprises and keeps
conversion Locality in one place; feature tests gain Leverage from a clear input
contract. Update examples and cover prepared values and invalid pixels before
changing behavior. Callers currently relying on automatic scaling must migrate.

## 6. Pipeline

### 12. Remove unfinished exports until a concrete caller needs them — small, breaking

**Files:** [pipeline/__init__.py](../../src/geosave_engine/geodata/pipeline/__init__.py),
[pipeline/publish.py](../../src/geosave_engine/geodata/pipeline/publish.py),
[pipeline/manifest.py](../../src/geosave_engine/geodata/pipeline/manifest.py),
`Manifest.update`.

**Evidence:** `push_to_hub`, `push_to_bucket`, and `Manifest.update` unconditionally
raise `NotImplementedError`. No current source/test caller of those operations
was found. Their docstrings describe behavior the Implementation does not have.

**Simplification:** These shallow Modules fail the deletion test: removing their
Interface removes unsupported promises without distributing useful behavior.
Keep the functioning Manifest index; add publishing through native dependencies
when a concrete use case defines it. This improves navigability and Locality,
and keeps the available Interface's Leverage honest. Add Manifest round-trip
tests for its existing behavior; there are no dedicated pipeline tests.

### 13. Enforce the manifest's portable-path invariant in one place — small

**Files:** [pipeline/manifest.py](../../src/geosave_engine/geodata/pipeline/manifest.py),
`_stored`, line 228.

**Evidence:** A path spelled `root / ".." / "elsewhere.tif"` is accepted and stored
as `../elsewhere.tif`, despite the documented requirement that files lie under
the manifest root. Copying the root cannot carry that referenced file with it.

**Simplification:** Normalize and validate containment at the existing storage
Seam, with a deliberate symlink policy. This gives the Manifest Module sole
Locality for the invariant and callers reliable Leverage from relative paths.
Test traversal, valid descendants, absolute/relative representations, symlinks,
save/reopen, and relocation. Keep multi-writer locking out of scope until there
is a real concurrent writer; current fixed-name staging is single-writer only.

## Checks and next choice

- `uv run pytest tests/geodata/attrs -q`: **72 passed**.
- `uv run pytest tests/geodata/stac -q`: **9 passed**, one warning.
- `uv run pytest tests/geodata/transform -q`: **185 passed**, five warnings.
- `uv run pytest tests/geodata -q`: **429 passed**, 31 warnings, 20.92 seconds.
  The complete run was outside the sandbox after a focused core run stalled in
  Zarr's background event loop. No external catalog or upload was used.
- Additional offline smokes reproduced the findings above. STAC decoding checks
  used the repository's local GeoTIFF fixture and the real ODC loading path.
  Separate exploration and independent code review checked the candidates.

In the requested module order, start with **1**, then **2**. The easiest isolated
performance improvement is **3**; the strongest pixel-correctness priorities are
**6** and **7**. Detailed replacement Interfaces have intentionally not been
designed yet. Existing passing tests do not cover the reproduced failures.

No library behavior changed in this sweep. Implementation choices involving
metadata serialization (**2**, **4**) and removal of public behavior (**11**,
**12**) need explicit migration decisions. Remote publishing, large-scene
accumulator storage, and broader workflow redesign were not evaluated as new
features in this review.
