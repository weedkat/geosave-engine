"""Historical backend comparison; its baseline uses native Tiler/Merger directly.

Run from the repository root. Artifacts and measurements are written to /tmp.
This is a full-parent NumPy prototype, not an out-of-core implementation.
"""

import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import torch
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from scipy.signal.windows import hann

from geosave_engine.geodata.io import geoparquet
from geosave_engine.geodata.transform.tiling import Tiles


def native_windows(shape, tile_shape, overlap):
    """Compare native layout arithmetic against Tiler, without using Tiler."""
    offsets = []
    for length, tile in zip(shape, tile_shape, strict=True):
        step = tile - overlap
        before = max(step // 2, overlap // 2) if overlap else 0
        after = before + step % 2 if overlap else 0
        count = (length + before + after - tile + step - 1) // step + 1
        offsets.append([position * step - before for position in range(count)])
    return [(row, col) for row in offsets[0] for col in offsets[1]]


class ReferenceAccumulator:
    """Probe state owns expected IDs, validity, completion, and parent buffers."""

    def __init__(self, reference, parents, window=None, leading_dims=("band",)):
        self.reference = reference.set_index("id", verify_integrity=True)
        self.parents = parents
        self.window = window
        self.leading_dims = leading_dims
        self.seen = set()
        self.states = {}
        self.completed = set()
        self.peak_bytes = 0

    @property
    def pending(self):
        return {
            parent: sum(key not in self.seen for key in rows.index)
            for parent, rows in self.reference.groupby("parent_id")
            if parent not in self.completed
            and any(key not in self.seen for key in rows.index)
        }

    def add(self, results, valid=None):
        # Validate the entire batch before changing completion or pixel buffers.
        batch = []
        leading_shapes = {
            name: state[0].shape[:-2] for name, state in self.states.items()
        }
        if valid is not None and set(valid) != set(results):
            raise ValueError("validity keys differ from results")
        for key, value in results.items():
            row = self.reference.loc[key]
            value = np.asarray(value)
            if key in self.seen:
                raise ValueError("duplicate tile")
            if value.shape[-2:] != (row.height, row.width):
                raise ValueError("output grid differs from tile")
            leading = value.shape[:-2]
            if len(leading) != len(self.leading_dims):
                raise ValueError("leading dimensions differ")
            if leading != leading_shapes.setdefault(row.parent_id, leading):
                raise ValueError("leading shapes differ")
            mask = np.isfinite(value).all(axis=tuple(range(value.ndim - 2)))
            if valid is not None:
                declared = np.asarray(valid[key])
                if declared.dtype != np.bool_ or declared.shape != (
                    row.height,
                    row.width,
                ):
                    raise ValueError("validity must be a spatial boolean mask")
                mask &= declared
            batch.append((key, row, value, mask))

        for key, row, value, mask in batch:
            parent = self.parents[row.parent_id]
            shape = tuple(parent.sizes[dim] for dim in parent.gs.grid_dims)
            if row.parent_id not in self.states:
                self.states[row.parent_id] = (
                    np.zeros((*value.shape[:-2], *shape), dtype=np.float64),
                    np.zeros(shape, dtype=np.float64),
                    value.dtype,
                )
            total, weight, _ = self.states[row.parent_id]
            y0, x0 = max(0, row.row_off), max(0, row.col_off)
            y1, x1 = (
                min(shape[0], row.row_off + row.height),
                min(shape[1], row.col_off + row.width),
            )
            source = (
                slice(y0 - row.row_off, y1 - row.row_off),
                slice(x0 - row.col_off, x1 - row.col_off),
            )
            target = (slice(y0, y1), slice(x0, x1))
            taper = (
                np.ones((row.height, row.width))
                if self.window is None
                else np.outer(hann(row.height, sym=False), hann(row.width, sym=False))
            )
            effective = taper[source] * mask[source]
            total[(..., *target)] += (
                np.where(mask[source], value[(..., *source)], 0) * effective
            )
            weight[target] += effective
            self.seen.add(key)
        self.peak_bytes = max(
            self.peak_bytes,
            sum(
                total.nbytes + weight.nbytes
                for total, weight, _ in self.states.values()
            ),
        )

    def merge(self):
        outputs = {}
        for parent_id, rows in self.reference.groupby("parent_id"):
            if parent_id in self.completed or not set(rows.index) <= self.seen:
                continue
            total, weight, dtype = self.states.pop(parent_id)
            pixels = np.full(total.shape, np.nan)
            np.divide(total, weight, out=pixels, where=weight > 0)
            parent = self.parents[parent_id]
            spatial = parent.gs.grid_dims
            coords = {
                name: value
                for name, value in parent.coords.items()
                if set(value.dims) <= set(spatial)
            }
            outputs[parent_id] = xr.DataArray(
                pixels.astype(dtype), dims=(*self.leading_dims, *spatial), coords=coords
            )
            self.completed.add(parent_id)
        return outputs

    def finish(self):
        if self.pending:
            raise ValueError(f"incomplete parents: {self.pending}")


def fixture(shape=(12, 16), geographic=True):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100
    grid = GeoBox(shape, Affine(10, 0, 600000, 0, -10, 9000000), "EPSG:32748")
    return xr.Dataset(
        {"B": (("y", "x"), values)}, coords=xr_coords(grid) if geographic else None
    )


def predictions(tiles, reference, invalid=False, leading=False):
    results = {}
    for position, row in enumerate(reference.itertuples()):
        pixels = tiles[position].B.values
        with torch.inference_mode():
            image = torch.as_tensor(pixels)
            value = torch.stack((image, image * 2, image * 0)).numpy()
        if leading:
            value = np.stack((value, value))
        if invalid:
            # Every contribution to parent pixel (3, 5) is invalid.
            yy, xx = 3 - row.row_off, 5 - row.col_off
            if 0 <= yy < row.height and 0 <= xx < row.width:
                value[(..., yy, xx)] = np.nan
        results[row.id] = value
    return results


def existing(tiles, reference, values, window, leading_dims=("band",)):
    """Reproduce the historical numeric baseline without a removed GeoSave API."""
    from tiler import Merger

    positions = {key: position for position, key in enumerate(reference.id)}
    states = {}
    leading = {}
    for key, value in values.items():
        ordinal, local = tiles.locate(positions[key])
        shape = value.shape[:-2]
        if ordinal not in states:
            states[ordinal] = Merger(
                tiles._cuts[ordinal].layout,
                window=window,
                logits=int(np.prod(shape)) if shape else 0,
                data_dtype=np.promote_types(value.dtype, np.float32),
                save_visits=False,
            )
            leading[ordinal] = shape
        states[ordinal].add(
            local, value.reshape(-1, *value.shape[-2:]) if shape else value
        )
    buffer_bytes = sum(
        state.data.nbytes + state.weights_sum.nbytes for state in states.values()
    )
    outputs = {}
    for ordinal, state in states.items():
        pixels = state.merge(unpad=True, extra_padding=tiles._cuts[ordinal].padding)
        outputs[ordinal] = xr.DataArray(
            pixels.reshape(*leading[ordinal], *pixels.shape[-2:]),
            dims=(*leading_dims, "y", "x"),
        )
    return outputs, buffer_bytes


def compare():
    root = Path(tempfile.mkdtemp(prefix="geosave-merger-comparison-"))
    results = {"fixtures": [], "benchmarks": [], "artifact_root": str(root)}
    parents = {
        "a": fixture(),
        "b": fixture().assign(B=lambda ds: ds.B * 0.5 + 1),
        "plain": fixture(geographic=False),
    }
    for tile_shape, overlap in [((8, 8), 0), ((8, 8), 4), ((6, 8), 2)]:
        tiles = Tiles(list(parents.values()), tile_shape, overlap=overlap)
        reference = tiles.reference(list(parents))
        path = geoparquet.write(
            reference,
            root / f"reference-{tile_shape[0]}-{overlap}.parquet",
            index=False,
        )
        shuffled = geoparquet.read(path).sample(frac=1, random_state=41)
        native = shuffled.copy()
        for parent_id, parent in parents.items():
            own = reference.loc[reference.parent_id == parent_id]
            offsets = native_windows(parent.B.shape, tile_shape, overlap)
            assert offsets == list(zip(own.row_off, own.col_off, strict=True))
            for key, (row_off, col_off) in zip(own.id, offsets, strict=True):
                native.loc[native.id == key, ["row_off", "col_off"]] = [
                    row_off,
                    col_off,
                ]

        for window in [None, "hann"] if overlap else [None]:
            for leading in [False, True]:
                dims = ("time", "band") if leading else ("band",)
                clean = predictions(tiles, reference, leading=leading)
                raw, _ = existing(tiles, reference, clean, window, dims)
                values = predictions(tiles, reference, invalid=True, leading=leading)
                np.savez(root / "raw.npz", **values)
                # Persistence and assembly do not depend on retaining the cut.
                with np.load(root / "raw.npz") as saved:
                    restored = {key: saved[key] for key in reversed(saved.files)}
                for table in [shuffled, native]:
                    accumulator = ReferenceAccumulator(table, parents, window, dims)
                    accumulator.add(restored)
                    for total, weight, _ in accumulator.states.values():
                        assert weight[3, 5] == 0
                        mask = np.ones(weight.shape, dtype=bool)
                        mask[3, 5] = False
                        assert np.all(weight[mask] > 0)
                    assembled = accumulator.merge()
                    accumulator.finish()
                    assert set(assembled) == set(parents)
                    for ordinal, parent_id in enumerate(parents):
                        actual = assembled[parent_id]
                        expected = raw[ordinal].values.copy()
                        pixels = parents[parent_id].B.values
                        direct = np.stack((pixels, pixels * 2, pixels * 0))
                        if leading:
                            direct = np.stack((direct, direct))
                        np.testing.assert_allclose(expected, direct, atol=1e-6)
                        expected[(..., 3, 5)] = np.nan
                        np.testing.assert_allclose(
                            actual.values, expected, atol=1e-6, equal_nan=True
                        )
                        assert actual.dims == (*dims, "y", "x")
                        assert actual.gs.geobox == parents[parent_id].gs.geobox
                        assert actual.dtype == np.float32
                        for dim in ("y", "x"):
                            if dim in parents[parent_id].coords:
                                xr.testing.assert_identical(
                                    actual.coords[dim], parents[parent_id].coords[dim]
                                )
                    try:
                        accumulator.add(
                            {next(iter(restored)): next(iter(restored.values()))}
                        )
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("duplicate after completion accepted")
                results["fixtures"].append(
                    {
                        "tile_shape": tile_shape,
                        "overlap": overlap,
                        "window": window,
                        "leading": leading,
                        "tiles": len(reference),
                    }
                )

        # Invalid batches must not partially update a valid preceding record.
        values = predictions(tiles, reference)
        first = next(iter(values))
        for bad in ["unknown", reference.id.iloc[1]]:
            accumulator = ReferenceAccumulator(shuffled, parents)
            malformed = {first: values[first], bad: np.zeros((3, 1, 1))}
            try:
                accumulator.add(malformed)
            except (ValueError, KeyError):
                assert accumulator.seen == set() and accumulator.states == {}
            else:
                raise AssertionError("invalid batch accepted")
        accumulator = ReferenceAccumulator(shuffled, parents)
        accumulator.add(
            {
                key: value
                for key, value in values.items()
                if shuffled.set_index("id").loc[key].parent_id != "b"
            }
        )
        assert accumulator.pending["b"] == sum(reference.parent_id == "b")
        try:
            accumulator.finish()
        except ValueError:
            pass
        else:
            raise AssertionError("wholly missing parent accepted")

    # A partially invalid overlapping contribution must not poison valid ones.
    parent = fixture()
    tiles = Tiles([parent], (8, 8), overlap=4)
    reference = tiles.reference(["a"])
    values = predictions(tiles, reference)
    first = reference.iloc[0]
    values[first.id][:, -1, -1] = np.nan
    accumulator = ReferenceAccumulator(reference, {"a": parent})
    accumulator.add(values)
    actual = accumulator.merge()["a"].values
    assert np.isfinite(actual).all()
    validity = {key: np.isfinite(value).all(axis=0) for key, value in values.items()}
    accumulator = ReferenceAccumulator(reference, {"a": parent})
    accumulator.add(
        {key: np.nan_to_num(value) for key, value in values.items()}, valid=validity
    )
    np.testing.assert_allclose(accumulator.merge()["a"].values, actual, atol=1e-6)
    old, _ = existing(tiles, reference, values, None)
    assert not np.allclose(old[0].values, actual, atol=1e-6)
    # Tiler normalizes NaN-contaminated sums to zero, despite valid neighbours.
    yy, xx = first.row_off + 7, first.col_off + 7
    results["invalid_overlap"] = {
        "existing": old[0].values[:, yy, xx].tolist(),
        "reference": actual[:, yy, xx].tolist(),
    }
    results["existing_validity_supported"] = False

    for length in [256, 512, 1024]:
        parent = fixture((length, length), geographic=False)
        tiles = Tiles([parent], (128, 128), overlap=32)
        reference = tiles.reference(["a"])
        values = predictions(tiles, reference)
        start = perf_counter()
        _, old_bytes = existing(tiles, reference, values, "hann")
        old_seconds = perf_counter() - start
        accumulator = ReferenceAccumulator(reference, {"a": parent}, "hann")
        start = perf_counter()
        accumulator.add(values)
        accumulator.merge()
        reference_seconds = perf_counter() - start
        start = perf_counter()
        offsets = native_windows((length, length), (128, 128), 32)
        assert offsets == list(zip(reference.row_off, reference.col_off, strict=True))
        native = reference.assign(
            row_off=[row for row, col in offsets], col_off=[col for row, col in offsets]
        )
        native_layout_seconds = perf_counter() - start
        accumulator = ReferenceAccumulator(native, {"a": parent}, "hann")
        start = perf_counter()
        accumulator.add(values)
        accumulator.merge()
        native_seconds = perf_counter() - start
        results["benchmarks"].append(
            {
                "shape": [length, length],
                "tiles": len(reference),
                "existing_seconds": old_seconds,
                "existing_buffer_bytes": old_bytes,
                "reference_seconds": reference_seconds,
                "reference_buffer_bytes": accumulator.peak_bytes,
                "native_seconds": native_seconds,
                "native_layout_seconds": native_layout_seconds,
                "native_buffer_bytes": accumulator.peak_bytes,
            }
        )
    results["recommended_candidate"] = (
        "Tiler windows with reference-driven accumulation"
    )
    results["bounded_spatial_memory"] = False
    (root / "report.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(compare(), indent=2))
