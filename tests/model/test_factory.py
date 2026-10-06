import pytest
from torch import nn

from geosave_engine.model.factory import BuildSpec


def test_imported_factory_preserves_missing_attribute_error() -> None:
    spec = BuildSpec(class_path="torch.nn.MissingLayer")

    with pytest.raises(AttributeError, match="MissingLayer"):
        spec.resolve({}, nn.Module)
