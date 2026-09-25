import dask.array as da
from dask.delayed import delayed
import numpy as np
from prefect.cache_policies import NO_CACHE
import pytest

from geosave_engine.geodata.utils import io
from geosave_engine.workflow.tasks.load import load_raster
from geosave_engine.workflow.tasks.save import save_stack


def test_native_results_are_not_cached_or_persisted():
    assert load_raster.cache_policy is NO_CACHE
    assert load_raster.persist_result is False
    assert save_stack.cache_policy is NO_CACHE
    assert save_stack.persist_result is False


def test_save_stack_completes_before_return_and_reopens(raw, tmp_path):
    output = tmp_path / "raw.zarr"

    saved = save_stack.fn(raw, output)

    assert saved == str(output)
    with io.read_stack(output, chunks="auto") as restored:
        assert restored.gs.groups == ("optical",)
        np.testing.assert_array_equal(restored["optical"].red, raw["optical"].red)


def test_save_stack_preserves_existing_destination(raw, tmp_path):
    output = tmp_path / "raw.zarr"
    save_stack.fn(raw, output)

    with pytest.raises(FileExistsError, match="already exists"):
        save_stack.fn({"replacement": raw["optical"] * 10}, output)

    with io.read_stack(output, chunks="auto") as restored:
        assert restored.gs.groups == ("optical",)
        np.testing.assert_array_equal(restored["optical"].red, raw["optical"].red)


def test_save_stack_rechecks_destination_before_publish(raw, tmp_path, monkeypatch):
    output = tmp_path / "raw.zarr"
    original = io.zarr.write

    def create_competing_output(*args, **kwargs):
        result = original(*args, **kwargs)
        output.mkdir()
        return result

    monkeypatch.setattr(io.zarr, "write", create_competing_output)

    with pytest.raises(FileExistsError, match="already exists"):
        save_stack.fn(raw, output)


def test_failed_write_cleans_staging_and_allows_retry(raw, tmp_path):
    def fail_chunk():
        raise RuntimeError("delayed chunk failed")

    failed = raw["optical"].copy()
    failed["red"] = (
        failed.red.dims,
        da.from_delayed(
            delayed(fail_chunk)(), shape=failed.red.shape, dtype=failed.red.dtype
        ),
    )
    output = tmp_path / "raw.zarr"

    with pytest.raises(RuntimeError, match="delayed chunk failed"):
        save_stack.fn({"optical": failed}, output)

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}-*")) == []
    assert save_stack.fn(raw, output) == str(output)
