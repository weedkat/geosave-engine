import pytest

from geosave_engine.ml.core.factory import builder, method_builder, uppercase_keys


class TestUppercaseKeys:
    def test_all_keys_uppercased(self):
        assert uppercase_keys({"a": 1, "Bb": 2}) == {"A": 1, "BB": 2}

    def test_empty(self):
        assert uppercase_keys({}) == {}

    def test_original_unchanged(self):
        d = {"x": 99}
        uppercase_keys(d)
        assert "x" in d


class _Dummy:
    def __init__(self, value: int = 0):
        self.value = value

    def make(self, multiplier: int = 1) -> int:
        return self.value * multiplier


REGISTRY = {"dummy": _Dummy}


class TestBuilder:
    def test_builds_registered_class(self):
        obj = builder("dummy", {"value": 5}, REGISTRY)
        assert isinstance(obj, _Dummy)
        assert obj.value == 5

    def test_case_insensitive_name(self):
        obj = builder("DUMMY", {}, REGISTRY)
        assert isinstance(obj, _Dummy)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown 'nope'"):
            builder("nope", {}, REGISTRY)

    def test_empty_config(self):
        obj = builder("dummy", {}, REGISTRY)
        assert obj.value == 0


class TestMethodBuilder:
    def test_calls_method(self):
        obj_registry = {"adam": _Dummy(value=3)}
        result = method_builder("adam.make", {"multiplier": 4}, obj_registry)
        assert result == 12

    def test_case_insensitive_key(self):
        obj_registry = {"adam": _Dummy(value=2)}
        result = method_builder("ADAM.make", {"multiplier": 1}, obj_registry)
        assert result == 2

    def test_missing_dot_raises(self):
        with pytest.raises(ValueError, match="Expected 'key.method' format"):
            method_builder("adam", {}, {"adam": _Dummy()})

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown key 'sgd'"):
            method_builder("sgd.make", {}, {"adam": _Dummy()})

    def test_unknown_method_raises(self):
        obj_registry = {"adam": _Dummy()}
        with pytest.raises(ValueError, match="Unknown method 'no_such'"):
            method_builder("adam.no_such", {}, obj_registry)
