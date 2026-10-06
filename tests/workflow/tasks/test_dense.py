import pandas as pd
import pytest
from prefect.cache_policies import NO_CACHE

from geosave_engine.model.spec import ModelSpec
from geosave_engine.workflow.tasks.dense import prepare_dense_sample, validate_row


def test_prepare_dense_sample_is_not_cached_or_persisted():
    assert prepare_dense_sample.cache_policy == NO_CACHE
    assert prepare_dense_sample.persist_result is False


def test_prepare_dense_sample_is_explicitly_pending(tmp_path):
    with pytest.raises(NotImplementedError, match="Dense sample catalog"):
        prepare_dense_sample.fn(
            tmp_path / "label.tif",
            ModelSpec(schema_version=2, rasters={}),
            tmp_path / "sample",
        )
    assert list(tmp_path.iterdir()) == []


def test_sample_validation_is_explicitly_pending():
    with pytest.raises(NotImplementedError, match="Dense sample catalog"):
        validate_row(pd.Series(dtype=object), ModelSpec(schema_version=2, rasters={}))
